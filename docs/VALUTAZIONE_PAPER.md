# Sistema di Valutazione Automatica dei Paper

Questo sistema permette di valutare automaticamente i paper utilizzando il sistema RAG multimodale, compilando automaticamente le risposte "local" nei file JSON di valutazione.

## Struttura dei File

### File di Valutazione JSON
I file devono seguire questa struttura:

```json
{
  "file": "nome_paper_07",
  "questions": [
    {
      "question_id": "q1",
      "question": "La tua domanda qui...",
      "Morphik": {
        "response": "...",
        "chunks": []
      },
      "local": {
        "response": "",
        "chunks": []
      }
    }
  ]
}
```

- **file**: Nome del paper che verrà usato per filtrare la ricerca nel database
- **questions**: Array di domande da valutare
- **local.response**: Verrà compilato automaticamente con la risposta del sistema RAG
- **local.chunks**: Verrà compilato con i numeri delle pagine utilizzate per la risposta

## Utilizzo

### 1. Preparazione dei File di Valutazione

**Opzione A: Crea Template Automaticamente**
```bash
cd /path/to/multimodalrag
python scripts/create_evaluation_templates.py
```
Questo crea 10 template base che puoi modificare con le tue domande.

**Opzione B: Crea Manualmente**
Metti i tuoi file JSON nella cartella `evaluation_data/` seguendo la struttura sopra.

### 2. Esecuzione della Valutazione Automatica

**Valuta tutti i file:**
```bash
python scripts/auto_evaluate_papers.py
```

**Valuta un singolo file:**
```bash
python scripts/auto_evaluate_papers.py --single-file example_paper_evaluation.json
```

**Sovrascrivi risposte esistenti:**
```bash
python scripts/auto_evaluate_papers.py --overwrite
```

**Mostra solo un report dei file disponibili:**
```bash
python scripts/auto_evaluate_papers.py --report
```

### 3. Configurazione Avanzata

**Directory personalizzata:**
```bash
python scripts/auto_evaluate_papers.py --eval-dir my_evaluations/
```

## Funzionamento

### Mappatura dei Nomi File
Il sistema converte automaticamente il nome del paper in possibili nomi di file nel database:

- `nome_paper_07` → `pdf_07.pdf`, `paper_07.pdf`, `document_07.pdf`, `07.pdf`
- `custom_name` → `custom_name.pdf`, `custom_name`

### Estrazione dei Chunk
Il sistema estrae automaticamente i numeri delle pagine utilizzate per generare la risposta, ordinandoli numericamente.

### Metadati Aggiuntivi
Per ogni risposta generata, vengono salvati metadati aggiuntivi:
- `confidence`: Livello di confidenza della risposta
- `retrieved_count`: Numero di documenti recuperati
- `file_filter_used`: Filtri file utilizzati
- `auto_generated`: Flag per indicare generazione automatica

## Output

### File Aggiornati
I file JSON vengono aggiornati in-place con le risposte e i chunk compilati.

### Report di Valutazione
Viene generato un report `evaluation_report.txt` con:
- Statistiche generali del processing
- Dettagli per ogni file processato
- Eventuali errori riscontrati

## Esempio di Output

```json
{
  "file": "nome_paper_07",
  "questions": [
    {
      "question_id": "q1",
      "question": "Qual è l'obiettivo principale del paper?",
      "Morphik": {
        "response": "...",
        "chunks": []
      },
      "local": {
        "response": "L'obiettivo principale del paper è...",
        "chunks": [1, 4, 6, 10, 12],
        "metadata": {
          "confidence": 0.8,
          "retrieved_count": 10,
          "file_filter_used": ["pdf_07.pdf"],
          "auto_generated": true
        }
      }
    }
  ]
}
```

## Risoluzione Problemi

### File Non Trovati
Se il sistema non trova documenti, verifica:
1. Che il database Qdrant sia in esecuzione
2. Che i documenti siano stati indicizzati correttamente
3. Che il nome del paper corrisponda ai file nel database

### Errori di Connessione
Assicurati che:
- Docker Qdrant sia avviato: `docker-compose up -d qdrant`
- Le variabili d'ambiente siano configurate correttamente
- Le dipendenze Python siano installate

### Risposte Vuote
Se le risposte sono vuote, potrebbe essere dovuto a:
- Threshold di similarità troppo alti
- Documenti non presenti nel database
- Query troppo specifiche o vaghe

## Monitoraggio

Durante l'esecuzione, il sistema fornisce logging dettagliato per:
- File processati
- Domande elaborate
- Chunk utilizzati per ogni risposta
- Eventuali errori o warning

Questo permette di monitorare il progresso e identificare eventuali problemi in tempo reale.
