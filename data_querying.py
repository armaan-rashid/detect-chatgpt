from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()
sql = """
SELECT CONCAT(questions.title, ' ', questions.body) AS question, answers.body AS answer
FROM bigquery-public-data.stackoverflow.posts_questions AS questions
INNER JOIN bigquery-public-data.stackoverflow.posts_answers AS answers
ON questions.accepted_answer_id = answers.id
WHERE questions.view_count > 5000;
"""
df = client.query(sql).to_dataframe()
df.to_json('data/full_questions_answers.json')
print(f"Received {len(df)} examples from BigQuery.")