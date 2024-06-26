(guardrails-data)=
# Guardrails for data management

Guardrails ensure intellectual property protection, safeguarding user privacy, alignment with legal and regulatory standards, and more. 
Mitigating these risks starts with the training data. If you train the model on private data, there's a good chance you'll get private data 
in the response. If you train a model on blogs that have toxic language or bias language towards different genders, you get the same results. 
The result will be the inability to trust the model’s results.

Data should be cleaned and prepared before it is sent to the model tuning or vector indexing process, for example, automatically removing PII. 
When collecting data, for example, you can identify PII automatically with the [PII recognizer function](https://www.mlrun.org/hub/functions/master/pii-recognizer/). 


