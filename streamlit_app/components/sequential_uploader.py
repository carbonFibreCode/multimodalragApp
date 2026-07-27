"""
Utilità per il caricamento sequenziale con timer e stato persistente.
"""

import time
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any

class SequentialUploader:
    """Gestisce il caricamento sequenziale di file con ritardi configurabili."""
    
    def __init__(self, delay_seconds: int = 30):
        """
        Inizializza l'uploader sequenziale.
        
        Args:
            delay_seconds: Ritardo in secondi tra i caricamenti
        """
        self.delay_seconds = delay_seconds
        self.session_keys = {
            'queue': 'seq_upload_queue',
            'in_progress': 'seq_upload_in_progress', 
            'results': 'seq_upload_results',
            'current_index': 'seq_upload_current_index',
            'start_time': 'seq_upload_start_time',
            'last_process_time': 'seq_upload_last_process_time'
        }
    
    def _init_session_state(self):
        """Inizializza le variabili di sessione se non esistono."""
        for key, session_key in self.session_keys.items():
            if session_key not in st.session_state:
                if key == 'queue':
                    st.session_state[session_key] = []
                elif key == 'results':
                    st.session_state[session_key] = []
                elif key in ['current_index']:
                    st.session_state[session_key] = 0
                elif key in ['in_progress']:
                    st.session_state[session_key] = False
                else:
                    st.session_state[session_key] = None
    
    def start_upload_sequence(self, files: List[Any]) -> bool:
        """
        Avvia la sequenza di caricamento.
        
        Args:
            files: Lista di file da caricare
            
        Returns:
            True se la sequenza è stata avviata
        """
        self._init_session_state()
        
        if st.session_state[self.session_keys['in_progress']]:
            return False
        
        st.session_state[self.session_keys['queue']] = files
        st.session_state[self.session_keys['in_progress']] = True
        st.session_state[self.session_keys['results']] = []
        st.session_state[self.session_keys['current_index']] = 0
        st.session_state[self.session_keys['start_time']] = time.time()
        st.session_state[self.session_keys['last_process_time']] = 0
        
        return True
    
    def stop_upload_sequence(self):
        """Ferma la sequenza di caricamento."""
        self._init_session_state()
        st.session_state[self.session_keys['in_progress']] = False
    
    def reset_upload_sequence(self):
        """Reset completo della sequenza."""
        for session_key in self.session_keys.values():
            if session_key in st.session_state:
                del st.session_state[session_key]
        self._init_session_state()
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Ottiene lo stato corrente della sequenza.
        
        Returns:
            Dizionario con informazioni sullo stato
        """
        self._init_session_state()
        
        queue = st.session_state[self.session_keys['queue']]
        current_index = st.session_state[self.session_keys['current_index']]
        start_time = st.session_state[self.session_keys['start_time']]
        last_process_time = st.session_state[self.session_keys['last_process_time']]
        
        status = {
            'in_progress': st.session_state[self.session_keys['in_progress']],
            'total_files': len(queue),
            'current_index': current_index,
            'completed_files': len(st.session_state[self.session_keys['results']]),
            'current_file': queue[current_index] if current_index < len(queue) else None,
            'is_finished': current_index >= len(queue),
            'results': st.session_state[self.session_keys['results']]
        }
        
        if start_time and not status['is_finished']:
            current_time = time.time()
            elapsed_since_start = current_time - start_time
            elapsed_since_last = current_time - last_process_time if last_process_time > 0 else elapsed_since_start
            
            # Tempo richiesto per il file corrente
            required_delay_for_current = current_index * self.delay_seconds
            
            status.update({
                'elapsed_time': elapsed_since_start,
                'required_delay': required_delay_for_current,
                'can_process_next': elapsed_since_start >= required_delay_for_current,
                'time_until_next': max(0, required_delay_for_current - elapsed_since_start),
                'estimated_completion': start_time + (len(queue) * self.delay_seconds)
            })
        
        return status
    
    def should_process_next_file(self) -> bool:
        """
        Verifica se è il momento di processare il prossimo file.
        
        Returns:
            True se si può processare il prossimo file
        """
        status = self.get_current_status()
        return (status['in_progress'] and 
                not status['is_finished'] and 
                status.get('can_process_next', False))
    
    def mark_file_processed(self, file_name: str, success: bool, message: str):
        """
        Marca un file come processato e aggiunge il risultato.
        
        Args:
            file_name: Nome del file processato
            success: True se il processing è riuscito
            message: Messaggio di risultato
        """
        self._init_session_state()
        
        result = {
            "file_name": file_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "index": st.session_state[self.session_keys['current_index']]
        }
        
        st.session_state[self.session_keys['results']].append(result)
        st.session_state[self.session_keys['current_index']] += 1
        st.session_state[self.session_keys['last_process_time']] = time.time()
        
        # Se tutti i file sono stati processati, ferma la sequenza
        if st.session_state[self.session_keys['current_index']] >= len(st.session_state[self.session_keys['queue']]):
            st.session_state[self.session_keys['in_progress']] = False
    
    def get_progress_percentage(self) -> float:
        """
        Calcola la percentuale di completamento.
        
        Returns:
            Percentuale da 0.0 a 1.0
        """
        status = self.get_current_status()
        if status['total_files'] == 0:
            return 1.0
        return status['current_index'] / status['total_files']
    
    def format_time_remaining(self, seconds: float) -> str:
        """
        Formatta il tempo rimanente in modo leggibile.
        
        Args:
            seconds: Secondi rimanenti
            
        Returns:
            Stringa formattata (es. "2m 30s")
        """
        if seconds <= 0:
            return "0s"
        
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Ottiene statistiche sui risultati.
        
        Returns:
            Dizionario con contatori di successi/fallimenti
        """
        results = st.session_state.get(self.session_keys['results'], [])
        
        return {
            'total': len(results),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success'])
        }
