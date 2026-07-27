import streamlit as st
import os
import time
import threading
from datetime import datetime
from src.config import RAW_DATA_PATH
from src.pipeline.retriever import enhanced_rag_query
from streamlit_app.backend_logic import process_uploaded_file, delete_source
from streamlit_app.components.sequential_uploader import SequentialUploader

def upload_widget(indexer):
    """Upload widget per più documenti con caricamento sequenziale."""
    st.header("Upload Documents")
    
    # Inizializza l'uploader sequenziale
    uploader = SequentialUploader(delay_seconds=30)
    
    # Caricamento file
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type="pdf", 
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Seleziona uno o più file PDF per il caricamento sequenziale con ritardo di 30 secondi"
    )

    # Ottieni lo stato corrente
    status = uploader.get_current_status()
    
    # Interfaccia per file non ancora in processing
    if uploaded_files and not status['in_progress']:
        st.info(f"📁 Selezionati {len(uploaded_files)} file per il caricamento")
        
        # Mostra l'elenco dei file
        with st.expander("File selezionati", expanded=True):
            for i, file in enumerate(uploaded_files, 1):
                file_size_mb = file.size / (1024 * 1024)
                st.write(f"{i}. **{file.name}** ({file_size_mb:.1f} MB)")
        
        # Controlli
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("Avvia Caricamento", type="primary"):
                if uploader.start_upload_sequence(uploaded_files):
                    st.success("Caricamento sequenziale avviato!")
                    st.rerun()
                else:
                    st.error("Errore nell'avvio del caricamento")
        
        with col2:
            st.info("Ritardo: 30s")
        
        with col3:
            estimated_time = len(uploaded_files) * 30
            st.info(f"Tempo stimato: {estimated_time//60}m {estimated_time%60}s")

    # Interfaccia durante il processing
    elif status['in_progress']:
        # Header con informazioni generali
        st.markdown(f"### Caricamento in corso...")
        
        # Progress bar principale
        progress = uploader.get_progress_percentage()
        st.progress(progress, text=f"File {status['current_index']}/{status['total_files']} completati")
        
        # Informazioni sul file corrente
        if not status['is_finished'] and status['current_file']:
            current_file = status['current_file']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"📄 **File corrente:** {current_file.name}")
                
                # Verifica se può processare
                if uploader.should_process_next_file():
                    # Processo il file corrente
                    upload_key = f"seq_processed_{current_file.name}_{current_file.size}_{status['current_index']}"
                    
                    if upload_key not in st.session_state:
                        with st.spinner(f"Indicizzando '{current_file.name}'..."):
                            success, message = process_uploaded_file(current_file, indexer)
                        
                        # Marca come processato
                        uploader.mark_file_processed(current_file.name, success, message)
                        st.session_state[upload_key] = True
                        
                        # Mostra risultato
                        if success:
                            st.success(f"Ok: {message}")
                        else:
                            st.error(f"Error: {message}")

                        st.rerun()
                else:
                    # Mostra countdown
                    time_remaining = status.get('time_until_next', 0)
                    if time_remaining > 0:
                        st.info(f"Prossimo file tra: {uploader.format_time_remaining(time_remaining)}")
                        time.sleep(1)
                        st.rerun()
            
            with col2:
                if st.button("Stop  ", type="secondary"):
                    uploader.stop_upload_sequence()
                    st.warning("Upload interrotto")
                    st.rerun()
        
        # Risultati dei file già processati
        if status['results']:
            with st.expander(f"Results ({len(status['results'])} processed files)", expanded=False):
                for result in status['results']:
                    status_icon = "Ok" if result["success"] else "Error"
                    st.write(f"{status_icon} **{result['file_name']}** - {result['timestamp']}")
                    if not result["success"]:
                        st.caption(f"Alert: {result['message']}")

        # Auto-refresh durante il processing
        if not status['is_finished']:
            # Usa un container per il refresh automatico
            time.sleep(0.5)
            st.rerun()

    # Interfaccia finale quando tutto è completato
    elif status['results'] and not status['in_progress']:
        st.success("Sequential upload completed!")

        # Statistiche finali
        stats = uploader.get_statistics()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Files", stats['total'])
        with col2:
            st.metric("Successes", stats['successful'])
        with col3:
            st.metric("Failures", stats['failed'])

        # Dettagli risultati
        if stats['total'] > 0:
            with st.expander("Details", expanded=False):
                for result in status['results']:
                    status_icon = "Ok" if result["success"] else "Error"
                    st.write(f"{status_icon} **{result['file_name']}** - {result['timestamp']}")
                    if not result["success"]:
                        st.caption(f"Alert: {result['message']}")

        # Controlli finali
        col1, col2 = st.columns(2)
        with col1:
            if st.button("New Upload", type="primary"):
                uploader.reset_upload_sequence()
                st.rerun()
        
        with col2:
            if stats['failed'] > 0:
                if st.button("Retry Failures", type="secondary"):
                    # Logica per riprovare solo i file falliti
                    failed_files = [r for r in status['results'] if not r['success']]
                    if failed_files:
                        st.info(f"Retrying {len(failed_files)} failed files...")
                        # Qui potresti implementare la logica per riprovare solo i fallimenti


def source_selector_widget():
    """Draw the widget to select and delete sources."""
    st.header("Manage and Select Sources")

    if not os.path.exists(RAW_DATA_PATH) or not os.listdir(RAW_DATA_PATH):
        st.info("No documents found. Please upload one to get started.")
        return []

    available_files = sorted([f for f in os.listdir(RAW_DATA_PATH) if f.endswith(".pdf")])
    
    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = available_files

    selected_now = st.multiselect(
        "Select documents for chat:",
        options=available_files,
        default=st.session_state.selected_files
    )
    st.session_state.selected_files = selected_now

    st.markdown("---")
    with st.expander("Delete Selected Sources", expanded=True):
        for file in available_files:
            if st.button(f"Delete '{file}'", key=f"del_{file}"):
                if file in st.session_state.selected_files:
                    st.session_state.selected_files.remove(file)
                with st.spinner(f"Deleting '{file}'..."):
                    success, message = delete_source(file)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    
    return selected_now


def display_source_document(doc_info: dict):
    """
    Display the information of a source document in a more precise and clean way.

    Args:
        doc_info: Dictionary with document information
        index: Document index
    """
    doc_type = doc_info.get("type", "text")
    source = doc_info.get("source", "Unknown Source")
    page = doc_info.get("page", "N/A")

    # Emoji and colors for document type
    type_info = {
        "text": {"emoji": "📝", "color": "#1f77b4"},
        "image": {"emoji": "🖼️", "color": "#2ca02c"}, 
        "table": {"emoji": "📊", "color": "#ff7f0e"}
    }
    
    info = type_info.get(doc_type, {"emoji": "📄", "color": "#7f7f7f"})
    
    # Header
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, {info["color"]}22 0%, {info["color"]}11 100%);
        border-left: 4px solid {info["color"]};
        padding: 12px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 0.9em;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <strong>{info["emoji"]} {source}</strong> - Page {page}
            </div>
            <span style="
                background-color: {info["color"]};
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 0.75em;
                font-weight: bold;
            ">{doc_type.upper()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def enhanced_chat_interface_widget(selected_sources: list[str]):
    """
    Enhanced chat interface with precise and clean references.
    """
    st.header("Chat with Selected Sources")

    if not selected_sources:
        st.warning("Please select at least one source to start the chat.")
        return

    show_sources = True
    compact_sources = True
    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Ask a question about the selected documents. Precise references will be shown below each response."}
        ]

    # Message display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show source documents in an enhanced way
            if (msg["role"] == "assistant" and 
                "source_docs" in msg and 
                show_sources and 
                msg["source_docs"]):
                
                st.markdown("---")
                
                if compact_sources:
                    # Compact view with precise references
                    st.markdown("**Source Documents:**")

                    # Group by source and page
                    sources_summary = {}
                    for doc in msg["source_docs"]:
                        if isinstance(doc, dict):
                            source = doc.get("source", "Unknown Source")
                            page = doc.get("page", "N/A")
                            doc_type = doc.get("type", "text")
                        else:
                            source = str(doc)
                            page = "N/A"
                            doc_type = "text"
                        
                        key = f"{source}"
                        if key not in sources_summary:
                            sources_summary[key] = {"pages": set(), "types": set()}
                        
                        sources_summary[key]["pages"].add(str(page))
                        sources_summary[key]["types"].add(doc_type)

                    # Show references in a compact and precise format
                    for source, info in sources_summary.items():
                        pages_list = sorted(info["pages"], key=lambda x: int(x) if x.isdigit() else float('inf'))
                        types_list = sorted(info["types"])

                        # Compact format: "document.pdf: p.1,2,3 (text, image)"
                        if len(pages_list) == 1:
                            pages_str = f"p.{pages_list[0]}"
                        else:
                            pages_str = f"pp.{','.join(pages_list)}"
                        
                        types_str = ", ".join(types_list)
                        
                        st.markdown(f"**{source}**: {pages_str} *({types_str})*")

    # User input and response generation
    if prompt := st.chat_input("Ask your question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = enhanced_rag_query(
                    query=prompt,
                    selected_files=selected_sources,
                )
                
                if not response:
                    answer = "Sorry, I couldn't find relevant information in the selected documents."
                    source_docs = []
                else:
                    answer = response.answer or "No response generated."
                    source_docs = response.source_documents or []

                # Show the answer directly
                edited_answer = answer
                st.markdown(edited_answer)

                # Save message with source documents
                assistant_message = {
                    "role": "assistant", 
                    "content": edited_answer,
                    "source_docs": source_docs if show_sources else []
                }
                st.session_state.messages.append(assistant_message)

                # Show immediate references
                if show_sources and source_docs:
                    st.markdown("---")
                    
                    # Limit source documents display to top 10 by score
                    display_docs = source_docs[:10] if len(source_docs) > 10 else source_docs
                    if len(source_docs) > 10:
                        st.info(f"Showing top 10 sources out of {len(source_docs)} total results")
                    
                    if compact_sources:
                        # Compact and precise references
                        st.markdown("**Origin sources:**")

                        sources_summary = {}
                        for doc in display_docs:  # Use limited documents
                            source = doc.get("source", "Unknown")
                            page = doc.get("page", "N/A")
                            doc_type = doc.get("type", "text")
                            
                            key = f"{source}"
                            if key not in sources_summary:
                                sources_summary[key] = {"pages": set(), "types": set()}
                            
                            sources_summary[key]["pages"].add(str(page))
                            sources_summary[key]["types"].add(doc_type)

                        # Show references in a compact and precise format
                        for source, info in sources_summary.items():
                            pages_list = sorted(info["pages"], key=lambda x: int(x) if x.isdigit() else float('inf'))
                            types_list = sorted(info["types"])

                            # More compact format: "📄 document.pdf: p.1,2,3 (text, image)"
                            if len(pages_list) == 1:
                                pages_str = f"p.{pages_list[0]}"
                            else:
                                pages_str = f"pp.{','.join(pages_list)}"
                            
                            types_str = ", ".join(types_list)
                            
                            # Emoji based on document type
                            if "image" in types_list:
                                emoji = "🖼️"
                            elif "table" in types_list:
                                emoji = "📊"
                            else:
                                emoji = "📄"
                            
                            st.markdown(f"{emoji} **{source}**: {pages_str} *({types_str})*")
                    
                    else:
                        # Detailed references
                        st.markdown("**Detailed origin sources:**")
                        for doc in display_docs:  # Use limited documents
                            display_source_document(doc)
                
                st.rerun()

