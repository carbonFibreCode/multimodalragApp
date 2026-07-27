#!/usr/bin/env python3
"""
Script per analizzare il tasso di correttezza del modello locale rispetto a Morphik (ground truth)
e identificare aree di forza e debolezza.
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usa backend non-interattivo per evitare errori macOS
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Dict, List, Tuple, Any

class CorrectnessAnalyzer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Inizializza l'analizzatore di correttezza.
        
        Args:
            model_name: Nome del modello SBERT da utilizzare
        """
        print(f"Caricamento del modello SBERT: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Soglie per la classificazione della correttezza
        self.thresholds = {
            'excellent': 0.85,
            'good': 0.70,
            'acceptable': 0.50,
            'poor': 0.30
        }
        
    def load_benchmark_data(self, directory: str = "./evaluation_data") -> List[Dict]:
        """Carica tutti i file di benchmark."""
        import os
        benchmark_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        benchmark_data = []
        
        for file in sorted(benchmark_files):
            file_path = os.path.join(directory, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        benchmark_data.extend(data)
                    else:
                        benchmark_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Errore nel parsing di {file}: {e}")
                    continue
        
        return benchmark_data
    
    def classify_correctness(self, similarity: float) -> str:
        """Classifica il livello di correttezza basato sulla similarità."""
        if similarity >= self.thresholds['excellent']:
            return 'Eccellente'
        elif similarity >= self.thresholds['good']:
            return 'Buono'
        elif similarity >= self.thresholds['acceptable']:
            return 'Accettabile'
        elif similarity >= self.thresholds['poor']:
            return 'Scarso'
        else:
            return 'Molto Scarso'
    
    def extract_difficulty_from_id(self, question_id: str) -> int:
        """Estrae livello di difficoltà dall'ID domanda (q1=1 → q10=10)."""
        try:
            # Estrae il numero dall'ID (es. "q3" → 3)
            difficulty = int(re.findall(r'\d+', question_id)[0])
            return min(max(difficulty, 1), 10)  # Assicura range 1-10
        except (IndexError, ValueError):
            return 5  # Difficoltà media di default
    
    def extract_macro_topic(self, question: str) -> str:
        """Estrae macro-argomento dalla domanda."""
        question_lower = question.lower()
        
        # Definizioni di macro-argomenti con parole chiave
        topic_keywords = {
            'Energia': ['energia', 'elettricità', 'potenza', 'watt', 'terawatt', 'elettrico', 
                       'produzione energetica', 'generazione', 'fattorie a onde', 'onde marine'],
            'Tecnologia': ['framework', 'modello', 'algoritmo', 'cnn', 'lstm', 'deep learning',
                          'neural', 'rete neurale', 'machine learning', 'ai', 'self-attention'],
            'Geografia': ['australia', 'sydney', 'adelaide', 'perth', 'tasmania', 'costa',
                         'località', 'meridionale', 'regione', 'area geografica'],
            'Matematica': ['percentuale', 'accuratezza', 'r2', 'prestazioni', 'metriche',
                          'calcolo', 'valore', 'numero', 'parametri', 'statistiche'],
            'Ambiente': ['sostenibile', 'sdg', 'climatico', 'green', 'neutralità', 'carbonio',
                        'ambientale', 'clima', 'sostenibilità', 'emissioni'],
            'Ingegneria': ['ottimizzazione', 'design', 'sistema', 'architettura', 'tecnico',
                          'implementazione', 'sviluppo', 'progettazione', 'costruzione'],
            'Economia': ['commercializzazione', 'investimenti', 'costi', 'economico', 'mercato',
                        'business', 'finanziario', 'commerciale', 'redditività'],
            'Fisica': ['onde', 'frequenza', 'ampiezza', 'fisica', 'meccanica', 'dinamica',
                      'oscillazione', 'fenomeno fisico', 'proprietà fisiche']
        }
        
        # Conta occorrenze per ogni topic
        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                topic_scores[topic] = score
        
        # Restituisce il topic con più matches, altrimenti "Generale"
        if topic_scores:
            return max(topic_scores.keys(), key=lambda k: topic_scores[k])
        else:
            return 'Generale'

    def extract_question_features(self, question: str, question_id: str = "") -> Dict[str, Any]:
        """Estrae caratteristiche dalla domanda per identificare pattern."""
        features = {}
        
        # Difficoltà dal question_id
        features['difficulty'] = self.extract_difficulty_from_id(question_id)
        features['difficulty_category'] = (
            'Facile' if features['difficulty'] <= 3 else
            'Media' if features['difficulty'] <= 7 else
            'Difficile'
        )
        
        # Macro-argomento
        features['macro_topic'] = self.extract_macro_topic(question)
        
        # Lunghezza della domanda
        features['question_length'] = len(question)
        features['word_count'] = len(question.split())
        
        # Tipo di domanda (basato su parole chiave)
        question_lower = question.lower()
        
        # Domande definitorie
        if any(word in question_lower for word in ['qual è', 'cos\'è', 'che cosa', 'definisci', 'spiega']):
            features['question_type'] = 'Definitoria'
        # Domande comparative
        elif any(word in question_lower for word in ['confronta', 'differenza', 'rispetto', 'paragonabile']):
            features['question_type'] = 'Comparativa'
        # Domande procedurali
        elif any(word in question_lower for word in ['come', 'in che modo', 'processo', 'metodo', 'descrivi']):
            features['question_type'] = 'Procedurale'
        # Domande analitiche
        elif any(word in question_lower for word in ['perché', 'causa', 'motivo', 'ragione', 'analizza']):
            features['question_type'] = 'Analitica'
        # Domande di enumerazione
        elif any(word in question_lower for word in ['quali', 'elenca', 'identifica', 'località', 'aspetti']):
            features['question_type'] = 'Enumerativa'
        else:
            features['question_type'] = 'Altra'
        
        return features
    
    def analyze_response_quality(self, morphik_response: str, local_response: Any) -> Dict[str, Any]:
        """Analizza la qualità della risposta del modello locale."""
        
        # Gestisce il caso in cui local_response potrebbe essere un dizionario (formato anomalo)
        if isinstance(local_response, dict):
            # Estrae il testo dalla struttura JSON anomala
            if 'response' in local_response:
                actual_response = str(local_response['response'])
            elif 'punti_principali' in local_response or 'conclusioni' in local_response:
                # Combina le sezioni strutturate in un testo unico
                parts = []
                if 'punti_principali' in local_response:
                    punti = local_response['punti_principali']
                    if isinstance(punti, dict):
                        parts.extend([str(v) for v in punti.values()])
                    else:
                        parts.append(str(punti))
                if 'conclusioni' in local_response:
                    parts.append(str(local_response['conclusioni']))
                if 'contenuto_multimodale' in local_response:
                    parts.append(str(local_response['contenuto_multimodale']))
                actual_response = ' '.join(parts)
            else:
                actual_response = str(local_response)
            local_response = actual_response
        else:
            local_response = str(local_response)
        
        # Gestisce risposte vuote o casi di non risposta
        if not local_response or local_response.strip() == "" or "non è stato in grado di rispondere" in local_response.lower():
            return {
                'semantic_similarity': 0.0,
                'correctness_level': 'Molto Scarso',
                'length_ratio': 0.0,
                'lexical_similarity': 0.0,
                'term_coverage': 0.0,
                'morphik_length': len(morphik_response),
                'local_length': 0,
                'morphik_terms': 0,
                'local_terms': 0
            }
        
        # Calcola similarità semantica
        embeddings = self.model.encode([morphik_response, local_response])
        similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
        
        # Assicura che la similarità sia nel range [0, 1]
        similarity = max(0.0, min(1.0, float(similarity)))
        
        # Analisi delle lunghezze
        morphik_len = len(morphik_response)
        local_len = len(local_response)
        length_ratio = local_len / morphik_len if morphik_len > 0 else 0
        
        # Assicura che il rapporto lunghezza sia ragionevole
        length_ratio = max(0.0, length_ratio)
        
        # Analisi del contenuto
        morphik_words = set(morphik_response.lower().split())
        local_words = set(local_response.lower().split())
        
        word_overlap = len(morphik_words.intersection(local_words))
        word_union = len(morphik_words.union(local_words))
        lexical_similarity = word_overlap / word_union if word_union > 0 else 0
        
        # Presenza di termini tecnici chiave
        key_terms = ['energia', 'elettricità', 'potenza', 'onde', 'framework', 'modello', 'algoritmo',
                    'cnn', 'lstm', 'deep learning', 'neural', 'ottimizzazione', 'parametri', 'accuratezza', 'dataset'
                    'sostenibile', 'sdg', 'climatico', 'carbonio', 'emissioni', 'fisica', 'frequenza', 'ampiezza',
                    'sostenibilità', 'ambientale', 'green', 'neutralità', 'clima', 'sostenibile', 'emissioni']
        
        morphik_terms = sum(1 for term in key_terms if term in morphik_response.lower())
        local_terms = sum(1 for term in key_terms if term in local_response.lower())
        
        term_coverage = local_terms / morphik_terms if morphik_terms > 0 else 0
        term_coverage = max(0.0, min(1.0, term_coverage))  # Assicura range [0, 1]
        
        return {
            'semantic_similarity': similarity,
            'correctness_level': self.classify_correctness(similarity),
            'length_ratio': length_ratio,
            'lexical_similarity': max(0.0, lexical_similarity),
            'term_coverage': term_coverage,
            'morphik_length': morphik_len,
            'local_length': local_len,
            'morphik_terms': morphik_terms,
            'local_terms': local_terms
        }
    
    def analyze_chunk_retrieval(self, morphik_chunks: List[int], local_chunks: List[int]) -> Dict[str, Any]:
        """Analizza la qualità del retrieval dei chunk."""
        
        # Gestisce il caso di chunk vuoti (quando il modello non risponde)
        if not local_chunks or len(local_chunks) == 0:
            return {
                'chunk_precision': 0.0,
                'chunk_recall': 0.0,
                'chunk_f1': 0.0,
                'chunk_jaccard': 0.0,
                'morphik_chunk_count': len(morphik_chunks) if morphik_chunks else 0,
                'local_chunk_count': 0,
                'chunk_difference': -len(morphik_chunks) if morphik_chunks else 0
            }
        
        # Normalizza i chunk per il confronto - assumendo che entrambi siano 0-based
        # Morphik usa 0-based e local usa 1-based, normalizziamo local
        local_chunks_for_comparison = []
        morphik_chunks_for_comparison = morphik_chunks if morphik_chunks else []
        
        # Verifica se i chunk locali sono 1-based (contengono valori > max morphik chunks)
        if local_chunks and morphik_chunks:
            max_morphik = max(morphik_chunks)
            min_local = min(local_chunks)
            
            # Se il minimo locale è > 0 e sembra essere 1-based, normalizza
            if min_local > 0 and max(local_chunks) > max_morphik:
                local_chunks_for_comparison = [c - 1 for c in local_chunks if isinstance(c, int) and c > 0]
                print(f"  - NORMALIZZAZIONE: chunk locali da 1-based a 0-based")
            else:
                local_chunks_for_comparison = [c for c in local_chunks if isinstance(c, int) and c >= 0]
        else:
            local_chunks_for_comparison = [c for c in local_chunks if isinstance(c, int) and c >= 0]
        
        morphik_set = set(morphik_chunks_for_comparison)
        local_set = set(local_chunks_for_comparison)
        
        intersection = len(morphik_set.intersection(local_set))
        union = len(morphik_set.union(local_set))
        
        precision = intersection / len(local_set) if len(local_set) > 0 else 0
        recall = intersection / len(morphik_set) if len(morphik_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        jaccard = intersection / union if union > 0 else 0
        
        return {
            'chunk_precision': max(0.0, precision),
            'chunk_recall': max(0.0, recall),
            'chunk_f1': max(0.0, f1),
            'chunk_jaccard': max(0.0, jaccard),
            'morphik_chunk_count': len(morphik_chunks_for_comparison),
            'local_chunk_count': len(local_chunks_for_comparison),
            'chunk_difference': len(local_chunks_for_comparison) - len(morphik_chunks_for_comparison)
        }
    
    def perform_correctness_analysis(self) -> pd.DataFrame:
        """Esegue l'analisi completa di correttezza."""
        
        benchmark_data = self.load_benchmark_data()
        results = []
        
        for paper_data in benchmark_data:
            paper_name = paper_data.get('file', 'unknown')
            print(f"Analisi del paper: {paper_name}")
            
            for question_data in paper_data.get('questions', []):
                try:
                    question_id = question_data.get('question_id', 'unknown')
                    question_text = question_data.get('question', '')
                    
                    morphik_response = question_data['Morphik']['response']
                    local_response_raw = question_data['local']['response']
                    
                    # Gestisce il caso anomalo dove local response è un oggetto strutturato
                    if isinstance(local_response_raw, dict):
                        print(f"  - ATTENZIONE: Rilevato formato strutturato anomalo per {question_id}")
                        print(f"    Sezioni trovate: {list(local_response_raw.keys())}")
                    
                    morphik_chunks = question_data['Morphik']['chunks']
                    local_chunks = question_data['local']['chunks']
                    
                    # Analisi delle caratteristiche della domanda
                    question_features = self.extract_question_features(question_text, question_id)
                    
                    # Analisi della qualità della risposta
                    response_analysis = self.analyze_response_quality(morphik_response, local_response_raw)
                    
                    # Analisi del retrieval dei chunk
                    chunk_analysis = self.analyze_chunk_retrieval(morphik_chunks, local_chunks)
                    
                    # Combina tutti i risultati
                    result = {
                        'paper': paper_name,
                        'question_id': question_id,
                        'question': question_text,
                        **question_features,
                        **response_analysis,
                        **chunk_analysis
                    }
                    
                    results.append(result)
                    
                    print(f"  - {question_id}: {response_analysis['correctness_level']} "
                          f"(Similarità: {response_analysis['semantic_similarity']:.3f})")
                    
                except Exception as e:
                    print(f"  - Errore nell'analisi della domanda {question_data.get('question_id', 'unknown')}: {e}")
                    continue
        
        return pd.DataFrame(results)
    
    def calculate_correctness_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcola metriche globali di correttezza."""
        
        total_questions = len(df)
        
        # Distribuzione per livello di correttezza
        correctness_dist = df['correctness_level'].value_counts()
        
        # Calcola tassi di correttezza
        excellent_rate = len(df[df['semantic_similarity'] >= self.thresholds['excellent']]) / total_questions
        good_plus_rate = len(df[df['semantic_similarity'] >= self.thresholds['good']]) / total_questions
        acceptable_plus_rate = len(df[df['semantic_similarity'] >= self.thresholds['acceptable']]) / total_questions
        
        # Statistiche generali
        mean_similarity = df['semantic_similarity'].mean()
        median_similarity = df['semantic_similarity'].median()
        std_similarity = df['semantic_similarity'].std()
        
        return {
            'total_questions': total_questions,
            'mean_similarity': mean_similarity,
            'median_similarity': median_similarity,
            'std_similarity': std_similarity,
            'excellent_rate': excellent_rate,
            'good_plus_rate': good_plus_rate,
            'acceptable_plus_rate': acceptable_plus_rate,
            'correctness_distribution': correctness_dist.to_dict()
        }
    
    def identify_failure_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identifica pattern di fallimento e successo."""
        
        # Soglia per considerare un fallimento
        failure_threshold = self.thresholds['acceptable']
        
        failures = df[df['semantic_similarity'] < failure_threshold]
        successes = df[df['semantic_similarity'] >= self.thresholds['good']]
        
        failure_analysis = {}
        success_analysis = {}
        
        # Analisi per tipo di domanda
        if len(failures) > 0:
            failure_analysis['by_question_type'] = failures['question_type'].value_counts().to_dict()
            failure_analysis['by_difficulty_category'] = failures['difficulty_category'].value_counts().to_dict()
            failure_analysis['by_macro_topic'] = failures['macro_topic'].value_counts().to_dict()
            failure_analysis['by_paper'] = failures['paper'].value_counts().to_dict()
            failure_analysis['avg_similarity'] = failures['semantic_similarity'].mean()
            failure_analysis['avg_length_ratio'] = failures['length_ratio'].mean()
            failure_analysis['avg_chunk_difference'] = failures['chunk_difference'].mean()
        
        if len(successes) > 0:
            success_analysis['by_question_type'] = successes['question_type'].value_counts().to_dict()
            success_analysis['by_difficulty_category'] = successes['difficulty_category'].value_counts().to_dict()
            success_analysis['by_macro_topic'] = successes['macro_topic'].value_counts().to_dict()
            success_analysis['by_paper'] = successes['paper'].value_counts().to_dict()
            success_analysis['avg_similarity'] = successes['semantic_similarity'].mean()
            success_analysis['avg_length_ratio'] = successes['length_ratio'].mean()
            success_analysis['avg_chunk_difference'] = successes['chunk_difference'].mean()
        
        return {
            'failures': failure_analysis,
            'successes': success_analysis,
            'failure_count': len(failures),
            'success_count': len(successes)
        }
    
    def create_essential_visualizations(self, df: pd.DataFrame, output_dir: str = "./evaluation_data/plots") -> None:
        """Crea grafici individuali separati per ogni analisi e salva i dati CSV corrispondenti."""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # GRAFICO 1: Performance per livello di difficoltà
        plt.figure(figsize=(10, 6))
        difficulty_perf = df.groupby('difficulty')['semantic_similarity'].agg(['mean', 'std', 'count'])
        
        # Salva dati CSV per Grafico 1
        difficulty_perf.to_csv(os.path.join(output_dir, '01_performance_per_difficolta_data.csv'))
        
        x_pos = difficulty_perf.index
        means = difficulty_perf['mean']
        stds = difficulty_perf['std']
        
        bars = plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
        plt.title('Performance per Livello di Difficoltà', fontsize=16, fontweight='bold')
        plt.xlabel('Livello Difficoltà (1=Facile → 10=Difficile)')
        plt.ylabel('Similarità Media (± Std Dev)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Aggiungi valori sopra le barre
        for bar, mean, count in zip(bars, means, difficulty_perf['count']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{mean:.2f}\n(n={count})', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '01_performance_per_difficolta.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # GRAFICO 2: Performance per Macro-Argomento
        plt.figure(figsize=(10, 8))
        topic_perf = df.groupby('macro_topic')['semantic_similarity'].agg(['mean', 'count']).sort_values('mean', ascending=True)
        
        # Salva dati CSV per Grafico 2
        topic_perf.to_csv(os.path.join(output_dir, '02_performance_per_argomento_data.csv'))
        
        y_pos = range(len(topic_perf))
        
        bars = plt.barh(y_pos, topic_perf['mean'], alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        plt.yticks(y_pos, topic_perf.index)
        plt.title('Performance per Macro-Argomento', fontsize=16, fontweight='bold')
        plt.xlabel('Similarità Media')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        
        # Aggiungi valori e conteggi
        for i, (mean, count) in enumerate(zip(topic_perf['mean'], topic_perf['count'])):
            plt.text(mean + 0.02, i, f'{mean:.2f} (n={count})', 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '02_performance_per_argomento.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # GRAFICO 3: Heatmap Difficoltà vs Macro-Argomento
        plt.figure(figsize=(10, 8))
        heatmap_data = df.pivot_table(values='semantic_similarity', 
                                     index='macro_topic', columns='difficulty_category', 
                                     aggfunc='mean')
        
        # Salva dati CSV per Grafico 3
        heatmap_data.to_csv(os.path.join(output_dir, '03_heatmap_argomento_difficolta_data.csv'))
        
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', fmt='.2f', 
                   cbar_kws={'label': 'Similarità Media'})
        plt.title('Heatmap: Argomento vs Difficoltà', fontsize=16, fontweight='bold')
        plt.xlabel('Categoria Difficoltà')
        plt.ylabel('Macro-Argomento')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '03_heatmap_argomento_difficolta.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # GRAFICO 4: Distribuzione complessiva con soglie
        plt.figure(figsize=(10, 6))
        similarity_values = df['semantic_similarity']
        
        # Salva dati CSV per Grafico 4
        similarity_stats = pd.DataFrame({
            'similarity_values': similarity_values,
            'correctness_level': df['correctness_level']
        })
        similarity_stats.to_csv(os.path.join(output_dir, '04_distribuzione_performance_data.csv'), index=False)
        
        # Gestisce range di valori appropriato per l'istogramma
        min_sim = similarity_values.min()
        max_sim = similarity_values.max()
        
        # Definisce bins appropriati basati sul range dei dati
        if min_sim < 0:
            bins = np.linspace(min_sim, max(max_sim, 1.0), 25)
        else:
            bins = np.linspace(0, max(max_sim, 1.0), 20)
        
        plt.hist(similarity_values, bins=bins, alpha=0.7, color='lightcoral', edgecolor='darkred')
        
        # Aggiungi linee di soglia solo se sono nel range visibile
        if max_sim >= self.thresholds['excellent']:
            plt.axvline(self.thresholds['excellent'], color='green', linestyle='--', linewidth=2,
                       label=f'Eccellente (≥{self.thresholds["excellent"]})')
        if max_sim >= self.thresholds['good']:
            plt.axvline(self.thresholds['good'], color='orange', linestyle='--', linewidth=2,
                       label=f'Buono (≥{self.thresholds["good"]})')
        if max_sim >= self.thresholds['poor']:
            plt.axvline(self.thresholds['poor'], color='red', linestyle='--', linewidth=2,
                       label=f'Soglia Critica (≥{self.thresholds["poor"]})')
        
        # Linea della media sempre visibile
        plt.axvline(similarity_values.mean(), color='blue', linestyle='-', linewidth=3,
                   label=f'Media: {similarity_values.mean():.3f}')
        
        # Aggiungi linea zero se ci sono valori negativi
        if min_sim < 0:
            plt.axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=2,
                       label='Zero')
        
        plt.title('Distribuzione Performance Globale', fontsize=16, fontweight='bold')
        plt.xlabel('Similarità Semantica')
        plt.ylabel('Frequenza')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '04_distribuzione_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # GRAFICO 5: Qualità del Retriever
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Subplot 1: Chunk Similarity vs Semantic Similarity (principale)
        scatter = ax1.scatter(df['chunk_jaccard'], df['semantic_similarity'], 
                             alpha=0.7, s=80, c=df['chunk_f1'], cmap='RdYlGn', 
                             edgecolors='black', linewidth=0.5)
        
        # Aggiungi linea di tendenza
        z = np.polyfit(df['chunk_jaccard'], df['semantic_similarity'], 1)
        p = np.poly1d(z)
        ax1.plot(df['chunk_jaccard'], p(df['chunk_jaccard']), "r--", alpha=0.8, linewidth=2)
        
        # Calcola correlazione
        correlation = df['chunk_jaccard'].corr(df['semantic_similarity'])
        ax1.text(0.05, 0.95, f'Correlazione: {correlation:.3f}', 
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
        
        ax1.set_xlabel('Similarità Chunk (Jaccard)', fontsize=12)
        ax1.set_ylabel('Similarità Semantica Risposte', fontsize=12)
        ax1.set_title('Chunk Similarity vs Qualità Risposte', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # Colorbar per F1 score
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('F1 Score Chunk', fontsize=10)
        
        # Subplot 2: Distribuzione qualità retriever
        # Crea categorie di qualità retriever
        df['retriever_quality'] = pd.cut(df['chunk_jaccard'], 
                                       bins=[0, 0.1, 0.3, 0.5, 1.0],
                                       labels=['Bassa (0-10%)', 'Media (10-30%)', 
                                              'Buona (30-50%)', 'Ottima (50%+)'])
        
        quality_counts = df['retriever_quality'].value_counts()
        colors = ['#ff4444', '#ffaa44', '#44aa44', '#44ff44']
        
        wedges, texts, autotexts = ax2.pie(quality_counts.values, 
                                          labels=quality_counts.index,
                                          colors=colors, autopct='%1.1f%%',
                                          startangle=90)
        
        ax2.set_title('Distribuzione Qualità Retriever\n(Similarità Chunk)', 
                     fontsize=14, fontweight='bold')
        
        # Aggiungi statistiche nel pie chart
        stats_text = f"""
        Media Jaccard: {df['chunk_jaccard'].mean():.3f}
        Media F1: {df['chunk_f1'].mean():.3f}
        Match perfetti: {len(df[df['chunk_jaccard'] == 1.0])}
        Nessuna overlap: {len(df[df['chunk_jaccard'] == 0.0])}
        """
        ax2.text(1.3, 0.5, stats_text, transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", facecolor='lightyellow', alpha=0.8),
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '05_qualita_retriever.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Salva dati CSV per Grafico 5 (semplificato)
        retriever_quality_data = df[['paper', 'question_id', 'chunk_jaccard', 'chunk_f1', 
                                   'chunk_precision', 'chunk_recall', 'semantic_similarity',
                                   'retriever_quality']].copy()
        
        # Statistiche aggregate
        retriever_stats = pd.DataFrame({
            'metric': ['jaccard_mean', 'jaccard_std', 'f1_mean', 'f1_std', 
                      'correlation_chunk_semantic', 'perfect_matches', 'no_overlap'],
            'value': [
                df['chunk_jaccard'].mean(), df['chunk_jaccard'].std(),
                df['chunk_f1'].mean(), df['chunk_f1'].std(),
                correlation,
                len(df[df['chunk_jaccard'] == 1.0]),
                len(df[df['chunk_jaccard'] == 0.0])
            ]
        })
        
        retriever_quality_data.to_csv(os.path.join(output_dir, '05_qualita_retriever_data.csv'), index=False)
        retriever_stats.to_csv(os.path.join(output_dir, '05_qualita_retriever_stats.csv'), index=False)
        plt.close()
        
        # GRAFICO 6: Analisi Lunghezza Risposte
        plt.figure(figsize=(12, 6))
        quality_bins = pd.cut(df['semantic_similarity'], bins=[0, 0.3, 0.5, 0.7, 0.85, 1.0], 
                             labels=['Molto Scarso', 'Scarso', 'Accettabile', 'Buono', 'Eccellente'])
        
        # Salva dati CSV per Grafico 6
        length_analysis = df[['semantic_similarity', 'length_ratio', 'question_id', 'paper']].copy()
        length_analysis['quality_category'] = quality_bins
        length_analysis.to_csv(os.path.join(output_dir, '06_lunghezza_per_qualita_data.csv'), index=False)
        
        box_data = [df[quality_bins == label]['length_ratio'].values for label in quality_bins.cat.categories if len(df[quality_bins == label]) > 0]
        valid_labels = [label for label in quality_bins.cat.categories if len(df[quality_bins == label]) > 0]
        
        if box_data:
            bp = plt.boxplot(box_data, labels=valid_labels, patch_artist=True)
            colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        plt.title('Rapporto Lunghezza per Qualità Risposta', fontsize=16, fontweight='bold')
        plt.ylabel('Rapporto Lunghezza (Local/Morphik)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '06_lunghezza_per_qualita.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # GRAFICO 7: Performance per Tipo di Domanda
        plt.figure(figsize=(12, 6))
        type_perf = df.groupby('question_type')['semantic_similarity'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        
        # Salva dati CSV per Grafico 7
        type_perf.to_csv(os.path.join(output_dir, '07_performance_per_tipo_domanda_data.csv'))
        
        bars = plt.bar(range(len(type_perf)), type_perf['mean'], 
                      alpha=0.7, color='mediumpurple', edgecolor='darkblue')
        plt.xticks(range(len(type_perf)), type_perf.index, rotation=45, ha='right')
        plt.title('Performance per Tipo di Domanda', fontsize=16, fontweight='bold')
        plt.ylabel('Similarità Media')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Aggiungi valori e conteggi
        for i, (mean, count) in enumerate(zip(type_perf['mean'], type_perf['count'])):
            plt.text(i, mean + 0.02, f'{mean:.2f}\n(n={count})', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '07_performance_per_tipo_domanda.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # GRAFICO 8: Coverage Termini Tecnici per Argomento
        plt.figure(figsize=(12, 6))
        term_coverage_data = df.groupby('macro_topic')['term_coverage'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        
        # Salva dati CSV per Grafico 8
        term_coverage_data.to_csv(os.path.join(output_dir, '08_coverage_termini_tecnici_data.csv'))
        
        bars = plt.bar(range(len(term_coverage_data)), term_coverage_data['mean'], 
                      yerr=term_coverage_data['std'], capsize=3,
                      alpha=0.7, color='gold', edgecolor='darkorange')
        plt.xticks(range(len(term_coverage_data)), term_coverage_data.index, rotation=45, ha='right')
        plt.title('Coverage Termini Tecnici per Argomento', fontsize=16, fontweight='bold')
        plt.ylabel('Coverage Media (± Std Dev)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Aggiungi valori
        for i, mean in enumerate(term_coverage_data['mean']):
            plt.text(i, mean + 0.02, f'{mean:.2f}', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '08_coverage_termini_tecnici.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # GRAFICO 10: Top 5 Peggiori Performance Effettive
        plt.figure(figsize=(12, 6))
        effective_responses = df[df['semantic_similarity'] != 0.0]
        if len(effective_responses) > 0:
            worst_effective = effective_responses.nsmallest(5, 'semantic_similarity')
            
            # Salva dati CSV per Grafico 10
            worst_effective[['question_id', 'paper', 'macro_topic', 'difficulty', 'semantic_similarity', 'question']].to_csv(
                os.path.join(output_dir, '10_peggiori_performance_data.csv'), index=False)
            
            case_labels = [f"{row['macro_topic'][:8]}\nD{row['difficulty']}\n{row['question_id']}" 
                          for _, row in worst_effective.iterrows()]
            
            bars = plt.bar(range(len(worst_effective)), worst_effective['semantic_similarity'], 
                          color='orange', alpha=0.7)
            plt.xticks(range(len(worst_effective)), case_labels, rotation=45, ha='right')
            plt.title(f'Top {len(worst_effective)} Peggiori Performance Effettive', fontsize=16, fontweight='bold')
            plt.ylabel('Similarità')
            
            # Imposta limiti y appropriati
            min_val = min(worst_effective['semantic_similarity'])
            max_val = max(worst_effective['semantic_similarity'])
            y_margin = 0.05
            plt.ylim(min(0, min_val - y_margin), max(max_val + y_margin, 0.5))
            
            # Aggiungi linea di riferimento se necessario
            if min_val < 0:
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            
            # Aggiungi valori
            for bar, sim in zip(bars, worst_effective['semantic_similarity']):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{sim:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'Tutti i casi sono\nnon-risposta', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=16, style='italic')
            plt.title('Peggiori Performance Effettive', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '10_peggiori_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # GRAFICO 11: Distribuzione per Categoria di Difficoltà (Boxplot)
        plt.figure(figsize=(10, 6))
        difficulty_cats = ['Facile', 'Media', 'Difficile']
        cat_data = []
        for cat in difficulty_cats:
            cat_df = df[df['difficulty_category'] == cat]
            if not cat_df.empty:
                cat_data.append(cat_df['semantic_similarity'].values)
            else:
                cat_data.append([])
        
        # Salva dati CSV per Grafico 11
        difficulty_boxplot_data = df[['difficulty_category', 'semantic_similarity', 'question_id', 'paper']].copy()
        difficulty_boxplot_data.to_csv(os.path.join(output_dir, '11_distribuzione_per_difficolta_data.csv'), index=False)
        
        # Rimuovi categorie vuote
        valid_cats = []
        valid_data = []
        for cat, data in zip(difficulty_cats, cat_data):
            if len(data) > 0:
                valid_cats.append(cat)
                valid_data.append(data)
        
        if valid_data:
            bp = plt.boxplot(valid_data, tick_labels=valid_cats, patch_artist=True)
            colors = ['lightgreen', 'yellow', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.title('Distribuzione per Difficoltà', fontsize=16, fontweight='bold')
            plt.ylabel('Similarità')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '11_distribuzione_per_difficolta.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # GRAFICO 12: Performance per Paper (se ci sono più paper)
        if df['paper'].nunique() > 1:
            plt.figure(figsize=(12, 6))
            paper_perf = df.groupby('paper')['semantic_similarity'].agg(['mean', 'count']).sort_values('mean', ascending=False)
            
            # Salva dati CSV per Grafico 12
            paper_perf.to_csv(os.path.join(output_dir, '12_performance_per_paper_data.csv'))
            
            bars = plt.bar(range(len(paper_perf)), paper_perf['mean'], 
                          alpha=0.7, color='teal', edgecolor='darkgreen')
            plt.xticks(range(len(paper_perf)), paper_perf.index, rotation=45, ha='right')
            plt.title('Performance per Paper/Documento', fontsize=16, fontweight='bold')
            plt.ylabel('Similarità Media')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            
            # Aggiungi valori e conteggi
            for i, (mean, count) in enumerate(zip(paper_perf['mean'], paper_perf['count'])):
                plt.text(i, mean + 0.02, f'{mean:.2f}\n(n={count})', 
                        ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '12_performance_per_paper.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # GRAFICO 13: Variabilità per Paper
            plt.figure(figsize=(14, 6))
            paper_data = []
            paper_labels = []
            for paper in df['paper'].unique():
                paper_df = df[df['paper'] == paper]
                if len(paper_df) > 1:  # Solo se ha più di una domanda
                    paper_data.append(paper_df['semantic_similarity'].values)
                    paper_labels.append(paper[:15])  # Abbrevia il nome
            
            # Salva dati CSV per Grafico 13
            if paper_data:
                variability_data = df[['paper', 'semantic_similarity', 'question_id']].copy()
                variability_data.to_csv(os.path.join(output_dir, '13_variabilita_per_paper_data.csv'), index=False)
            
            if paper_data:
                bp = plt.boxplot(paper_data, tick_labels=paper_labels, patch_artist=True)
                colors = sns.color_palette("husl", len(bp['boxes']))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                plt.title('Variabilità Performance per Paper', fontsize=16, fontweight='bold')
                plt.ylabel('Similarità')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '13_variabilita_per_paper.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(" Grafici individuali generati:")
        print(f"    {output_dir}/01_performance_per_difficolta.png")
        print(f"    {output_dir}/02_performance_per_argomento.png")
        print(f"    {output_dir}/03_heatmap_argomento_difficolta.png")
        print(f"    {output_dir}/04_distribuzione_performance.png")
        print(f"    {output_dir}/05_qualita_retriever.png")
        print(f"    {output_dir}/06_lunghezza_per_qualita.png")
        print(f"    {output_dir}/07_performance_per_tipo_domanda.png")
        print(f"    {output_dir}/08_coverage_termini_tecnici.png")
        print(f"    {output_dir}/10_peggiori_performance.png")
        print(f"    {output_dir}/11_distribuzione_per_difficolta.png")
        if df['paper'].nunique() > 1:
            print(f"    {output_dir}/12_performance_per_paper.png")
            print(f"    {output_dir}/13_variabilita_per_paper.png")
        
        print("\n Dati CSV corrispondenti salvati:")
        print(f"    {output_dir}/01_performance_per_difficolta_data.csv")
        print(f"    {output_dir}/02_performance_per_argomento_data.csv")
        print(f"    {output_dir}/03_heatmap_argomento_difficolta_data.csv")
        print(f"    {output_dir}/04_distribuzione_performance_data.csv")
        print(f"    {output_dir}/05_correlazione_chunk_performance_data.csv")
        print(f"    {output_dir}/06_lunghezza_per_qualita_data.csv")
        print(f"    {output_dir}/07_performance_per_tipo_domanda_data.csv")
        print(f"    {output_dir}/08_coverage_termini_tecnici_data.csv")
        print(f"    {output_dir}/10_peggiori_performance_data.csv")
        print(f"    {output_dir}/11_distribuzione_per_difficolta_data.csv")
        if df['paper'].nunique() > 1:
            print(f"    {output_dir}/12_performance_per_paper_data.csv")
            print(f"    {output_dir}/13_variabilita_per_paper_data.csv")

def main():
    """Funzione principale per l'analisi di correttezza."""
    
    print("Avvio Analisi di Correttezza del Modello Locale")
    print("=" * 60)
    
    # Inizializza l'analizzatore
    analyzer = CorrectnessAnalyzer()
    
    # Esegue l'analisi completa
    print("\nEsecuzione analisi di correttezza...")
    df = analyzer.perform_correctness_analysis()
    
    if df.empty:
        print("Nessun dato trovato per l'analisi!")
        return
    
    # Calcola metriche di correttezza
    print(f"\nAnalisi completata su {len(df)} domande")
    metrics = analyzer.calculate_correctness_metrics(df)
    
    print(f"\nRISULTATI PRINCIPALI:")
    print(f"   Similarità media: {metrics['mean_similarity']:.3f}")
    print(f"   Tasso Eccellente: {metrics['excellent_rate']:.1%}")
    print(f"   Tasso Buono+: {metrics['good_plus_rate']:.1%}")
    print(f"   Tasso Accettabile+: {metrics['acceptable_plus_rate']:.1%}")
    
    # Genera visualizzazioni
    print("\nGenerazione visualizzazioni...")
    analyzer.create_essential_visualizations(df)
    
    
    print(f"\nFile salvati:")
    print(f"   - correctness_analysis_detailed.csv")
    print(f"   - CORRECTNESS_REPORT.md")
    print(f"   - plots/01_performance_per_difficolta.png + CSV")
    print(f"   - plots/02_performance_per_argomento.png + CSV")
    print(f"   - plots/03_heatmap_argomento_difficolta.png + CSV")
    print(f"   - plots/04_distribuzione_performance.png + CSV")
    print(f"   - plots/05_qualita_retriever.png + CSV")
    print(f"   - plots/06_lunghezza_per_qualita.png + CSV")
    print(f"   - plots/07_performance_per_tipo_domanda.png + CSV")
    print(f"   - plots/08_coverage_termini_tecnici.png + CSV")
    print(f"   - plots/10_peggiori_performance.png + CSV")
    print(f"   - plots/11_distribuzione_per_difficolta.png + CSV")
    if df['paper'].nunique() > 1:
        print(f"   - plots/12_performance_per_paper.png + CSV")
        print(f"   - plots/13_variabilita_per_paper.png + CSV")
    
    print(f"\nAnalisi di correttezza completata!")


if __name__ == "__main__":
    main()
