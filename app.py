from flask import Flask, jsonify, request
from classifier import loadModel, classify
import boto3
import time
import csv
import os

model = loadModel()

app = Flask(__name__)

database = "nutrition_chatbot_db"
output = "s3://nutrition-chatbot-dev/athena-results"
table = "final_cleaned"
deficientNumbs={
    'iron' : 2,
    'vitamin_c' : 20,
    'calcium' : 100,
    'protein' : 10
}
def athenaQuery(query):
    client = boto3.client('athena', region_name='us-east-1')
    response = client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': database},
        ResultConfiguration={'OutputLocation': output}
    )
    queryID = response['QueryExecutionId']

    while True:
        result = client.get_query_execution(QueryExecutionId=queryID)
        state = result['QueryExecution']['Status']['State']
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(1)

    if state != 'SUCCEEDED':
        return []

    s3 = boto3.client('s3')
    bucket = "nutrition-chatbot-dev"
    key = f"athena-results/{queryID}.csv"
    file = "/tmp/nutrition_cleaned_recovered.csv"
    s3.download_file(bucket, key, file)

    with open(file, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


@app.route('/')
def home():
    return  "AI Nutrition Chatbot backend is running!"

@app.route('/recommend', methods=['GET'])
def recommend():
    deficiencies = request.args.getlist('def')
    conditions = []

    for nutrient in deficiencies:
        if nutrient in deficientNumbs:
            conditions.append(f"{nutrient} > {deficientNumbs[nutrient]}")

    if not conditions:
        return jsonify({"error": "No valid nutrient deficiencies provided"}), 400

    whereClause = " AND ".join(conditions)
    selectClause = ", ".join(['description'] + deficiencies)
    query = f"""
        SELECT {selectClause}
        FROM {table}
        WHERE {whereClause}
        ORDER BY {deficiencies[0]} DESC
        LIMIT 5;
    """

    results = athenaQuery(query)
    for meal in results:
        meal['classification'] = classify(meal, model)
    return jsonify({"recommendations": results})

if __name__ == '__main__':
    app.run(debug=True)
