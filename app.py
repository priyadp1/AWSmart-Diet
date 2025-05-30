from flask import Flask, request, jsonify
from classifier import predict_deficiency
import boto3
import time

app = Flask(__name__)

# Athena setup
DATABASE = "nutrition_chatbot_db"  # 
OUTPUT_LOCATION = "s3://nutrition-chatbot-dev/athena-results/"  
athena_client = boto3.client("athena", region_name="us-east-1") 

# Map human deficiency names to table column names
deficiency_column_map = {
    "vitamin c": "vitamin_c",
    "vitamin d": "vitamin_c", 
    "iron": "iron",
    "calcium": "calcium",
    "protein": "protein",
    "carbohydrate": "carbohydrate",
    "fat": "fat_total_lipid"
}

def query_athena(deficiency):
    column = deficiency_column_map.get(deficiency.lower())
    if not column:
        return []

    query = f"""
        SELECT description, {column}
        FROM {DATABASE}.final_cleaned
        WHERE {column} IS NOT NULL
        ORDER BY CAST({column} AS DOUBLE) DESC
        LIMIT 10
    """

    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": DATABASE},
        ResultConfiguration={"OutputLocation": OUTPUT_LOCATION}
    )
    query_execution_id = response['QueryExecutionId']

    while True:
        result = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        status = result['QueryExecution']['Status']['State']
        if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(1)

    if status == 'SUCCEEDED':
        results = athena_client.get_query_results(QueryExecutionId=query_execution_id)
        rows = results['ResultSet']['Rows'][1:]  # Skip header
        return [
            {
                "food": row['Data'][0].get('VarCharValue', ''),
                "value": row['Data'][1].get('VarCharValue', '')
            }
            for row in rows if len(row['Data']) >= 2
        ]
    else:
        return []

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.json.get('text', '')
    deficiency = predict_deficiency(user_input)
    recommendations = query_athena(deficiency)

    return jsonify({
        "deficiency": deficiency,
        "recommendations": recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)
