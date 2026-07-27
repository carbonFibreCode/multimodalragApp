#!/usr/bin/env python3
"""
Script per valutazione automatica dei paper.
Interroga il database RAG e compila automaticamente le risposte "local" nei file JSON di valutazione.
"""

import json
import sys
import glob
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Aggiungi il percorso del progetto al sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import del sistema RAG
from src.pipeline.retriever import enhanced_rag_query

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperEvaluator:
    """Classe per la valutazione automatica dei paper usando il sistema RAG."""
    
    def __init__(self, evaluation_files_dir: str = "evaluation_data"):
        """
        Inizializza il valutatore.
        
        Args:
            evaluation_files_dir: Directory contenente i file JSON di valutazione
        """
        self.evaluation_files_dir = evaluation_files_dir
        self.project_root = Path(__file__).parent.parent
        self.eval_dir_path = self.project_root / evaluation_files_dir
        
        # Crea la directory se non esiste
        self.eval_dir_path.mkdir(exist_ok=True)
        
    def find_evaluation_files(self) -> List[Path]:
        """Trova tutti i file JSON di valutazione nella directory."""
        pattern = str(self.eval_dir_path / "*.json")
        files = glob.glob(pattern)
        logger.info(f"Trovati {len(files)} file di valutazione in {self.eval_dir_path}")
        return [Path(f) for f in files]
    
    def load_evaluation_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Carica un file JSON di valutazione."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Caricato file: {file_path.name}")
            return data
        except Exception as e:
            logger.error(f"Errore nel caricamento di {file_path}: {e}")
            return None
    
    def save_evaluation_file(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """Salva un file JSON di valutazione."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Salvato file: {file_path.name}")
            return True
        except Exception as e:
            logger.error(f"Errore nel salvataggio di {file_path}: {e}")
            return False
    
    def extract_file_filter(self, file_name: str) -> List[str]:
        """
        Estrae il nome del file per il filtro dal nome del paper.
        
        Args:
            file_name: Nome del file dal JSON (es. "nome_paper_07")
        
        Returns:
            Lista di possibili nomi di file per il filtro
        """
        # Rimuovi "nome_paper_" e mantieni solo il numero
        if file_name.startswith("nome_paper_"):
            paper_num = file_name.replace("nome_paper_", "")
            # Genera possibili varianti del nome file
            possible_names = [
                f"pdf_{paper_num}.pdf",
                f"paper_{paper_num}.pdf", 
                f"document_{paper_num}.pdf",
                f"{paper_num}.pdf"
            ]
            return possible_names
        else:
            # Se non segue il pattern, usa il nome così com'è
            return [f"{file_name}.pdf", file_name]
    
    def query_rag_system(self, question: str, file_filter: List[str]) -> Dict[str, Any]:
        """
        Interroga il sistema RAG con una domanda e filtro file.
        
        Args:
            question: Domanda da porre al sistema
            file_filter: Lista di nomi file per filtrare la ricerca
        
        Returns:
            Dizionario con risposta e chunk utilizzati
        """
        try:
            logger.info(f"Querying RAG: '{question[:100]}...' with files: {file_filter}")
            
            # Esegui la query RAG
            result = enhanced_rag_query(
                query=question,
                selected_files=file_filter
            )
            
            # Estrai i chunk (pagine) utilizzati - MANTIENI L'ORDINE DI SIMILARITÀ
            chunks_used = []
            if result.source_documents:
                for doc in result.source_documents:
                    page = doc.get('page', 'N/A')
                    if page != 'N/A':
                        try:
                            page_int = int(page)
                            # Mantieni tutti i chunk, anche duplicati (poi rimuoveremo duplicati preservando ordine)
                            if page_int not in chunks_used:
                                chunks_used.append(page_int)
                        except (ValueError, TypeError):
                            # Se la pagina non è un numero, ignora
                            logger.warning(f"Pagina non numerica ignorata: {page}")
                            pass
            
            logger.info(f"Chunks estratti: {chunks_used} (da {len(result.source_documents)} documenti)")
            
            return {
                "response": result.answer,
                "chunks": chunks_used,
                "confidence": result.confidence_score,
                "retrieved_count": result.retrieved_count
            }
            
        except Exception as e:
            logger.error(f"Errore nella query RAG: {e}")
            return {
                "response": f"Errore nel recupero della risposta: {str(e)}",
                "chunks": [],
                "confidence": 0.0,
                "retrieved_count": 0
            }
    
    def process_single_file(self, file_path: Path, overwrite_existing: bool = False) -> bool:
        """
        Processa un singolo file di valutazione.
        
        Args:
            file_path: Percorso del file da processare
            overwrite_existing: Se True, sovrascrive le risposte esistenti
        
        Returns:
            True se il processing è andato a buon fine
        """
        # Carica il file
        data = self.load_evaluation_file(file_path)
        if not data:
            return False
        
        file_name = data.get('file', '')
        if not file_name:
            logger.warning(f"Nome file mancante in {file_path.name}")
            return False
        
        # Estrai il filtro per i file
        file_filter = self.extract_file_filter(file_name)
        logger.info(f"Usando filtro file: {file_filter} per paper: {file_name}")
        
        # Processa ogni domanda
        questions = data.get('questions', [])
        updated = False
        
        for i, question_data in enumerate(questions):
            question = question_data.get('question', '')
            local_data = question_data.get('local', {})
            
            # COMPORTAMENTO DESIDERATO: Processa sempre tutte le domande
            # Rimuovo il controllo per saltare domande già compilate
            
            if not question:
                logger.warning(f"Domanda vuota al index {i+1}")
                continue
            
            logger.info(f"Processando domanda {i+1}/{len(questions)}: {question[:80]}...")
            
            # Interroga il sistema RAG
            rag_result = self.query_rag_system(question, file_filter)
            
            # Aggiorna i dati locali
            local_data['response'] = rag_result['response']
            local_data['chunks'] = rag_result['chunks']
            
            # Aggiungi metadati opzionali
            if 'metadata' not in local_data:
                local_data['metadata'] = {}
            
            local_data['metadata'].update({
                'confidence': rag_result['confidence'],
                'retrieved_count': rag_result['retrieved_count'],
                'file_filter_used': file_filter,
                'auto_generated': True
            })
            
            updated = True
            
            logger.info(f"Completata domanda {i+1} - Chunk usati: {rag_result['chunks']}")
        
        # Salva il file se aggiornato
        if updated:
            return self.save_evaluation_file(file_path, data)
        else:
            logger.info(f"Nessun aggiornamento necessario per {file_path.name}")
            return True
    
    def process_all_files(self, overwrite_existing: bool = False) -> Dict[str, Any]:
        """
        Processa tutti i file di valutazione nella directory.
        
        Args:
            overwrite_existing: Se True, sovrascrive le risposte esistenti
        
        Returns:
            Statistiche del processing
        """
        files = self.find_evaluation_files()
        
        if not files:
            logger.warning("Nessun file di valutazione trovato!")
            return {"processed": 0, "successful": 0, "failed": 0}
        
        stats = {"processed": 0, "successful": 0, "failed": 0, "files": []}
        
        for file_path in files:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processando: {file_path.name}")
            logger.info(f"{'='*60}")
            
            stats["processed"] += 1
            
            try:
                success = self.process_single_file(file_path, overwrite_existing)
                if success:
                    stats["successful"] += 1
                    stats["files"].append({"file": file_path.name, "status": "success"})
                else:
                    stats["failed"] += 1
                    stats["files"].append({"file": file_path.name, "status": "failed"})
            except Exception as e:
                logger.error(f"Errore nel processing di {file_path.name}: {e}")
                stats["failed"] += 1
                stats["files"].append({"file": file_path.name, "status": "error", "error": str(e)})
        
        return stats
    
    def create_summary_report(self, stats: Dict[str, Any]) -> str:
        """Crea un report riassuntivo del processing."""
        report = f"""
REPORT VALUTAZIONE AUTOMATICA PAPER
{'='*50}

Statistiche Generali:
- File processati: {stats['processed']}
- Successo: {stats['successful']}
- Falliti: {stats['failed']}
- Tasso di successo: {(stats['successful']/stats['processed']*100) if stats['processed'] > 0 else 0:.1f}%

Dettagli per File:
"""
        
        for file_info in stats.get('files', []):
            status_emoji = "ok" if file_info['status'] == 'success' else "error"
            report += f"{status_emoji} {file_info['file']} - {file_info['status']}\n"
            if 'error' in file_info:
                report += f"    Errore: {file_info['error']}\n"
        
        return report


def main():
    """Funzione principale dello script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Valutazione automatica dei paper usando RAG")
    parser.add_argument("--eval-dir", default="evaluation_data", 
                       help="Directory contenente i file JSON di valutazione")
    parser.add_argument("--overwrite", action="store_true",
                       help="Sovrascrivi le risposte esistenti")
    parser.add_argument("--single-file", type=str,
                       help="Processa solo un singolo file specificato")
    parser.add_argument("--report", action="store_true",
                       help="Genera solo un report senza processare")
    
    args = parser.parse_args()
    
    # Inizializza il valutatore
    evaluator = PaperEvaluator(args.eval_dir)
    
    if args.report:
        # Solo report
        files = evaluator.find_evaluation_files()
        print(f"Trovati {len(files)} file di valutazione:")
        for f in files:
            print(f"  - {f.name}")
        return
    
    if args.single_file:
        # Processa un singolo file
        file_path = evaluator.eval_dir_path / args.single_file
        if not file_path.exists():
            logger.error(f"File non trovato: {file_path}")
            return
        
        success = evaluator.process_single_file(file_path, args.overwrite)
        if success:
            print(f"ok File {args.single_file} processato con successo")
        else:
            print(f"Errore nel processing di {args.single_file}")
    else:
        # Processa tutti i file
        print("Avvio valutazione automatica dei paper...")
        
        stats = evaluator.process_all_files(args.overwrite)
        report = evaluator.create_summary_report(stats)
        
        print(report)
        
        # Salva il report
        report_file = evaluator.eval_dir_path / "evaluation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\nReport salvato in: {report_file}")


if __name__ == "__main__":
    main()
