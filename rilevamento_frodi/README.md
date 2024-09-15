# Rilevamento delle frodi nei pagamenti online
Il dataset contiene i seguenti dati relativi alle transazioni di pagamento on line:

- step: rappresenta un'unità di tempo in cui 1 step equivale a 1 ora
- tipo: tipo di transazione online
- importo: l'importo della transazione
- nameOrig: cliente che avvia la transazione
- oldbalanceOrg: saldo prima della transazione
- newbalanceOrig: saldo dopo la transazione
- nameDest: destinatario della transazione
- oldbalanceDest: saldo iniziale del destinatario prima della transazione
- newbalanceDest: il nuovo saldo del destinatario dopo la transazione
- isFraud: transazione fraudolenta

*Costruire un modello di predizione in grado di prevedere se una transazione è una frode o no.*

In particolare:

1. Analizzare i dati a vostra disposizione e correggere eventuali errori presenti nell'insieme di dati
(dati mancanti, dati palesemente errati, etc...)
2. Individuare quali sono gli attributi che sembrano maggiormente correlati alla variabile target
3. Costruire almeno un paio di modelli previsivi, appartenenti a tipologie diverse (ad esempio alberi
di classificazione e classificatori a regole)
4. Studiare l'accuratezza dei modelli ottenuti