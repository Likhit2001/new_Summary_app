import json
import boto3

runtime = boto3.client("sagemaker-runtime")

def lambda_handler(event, context):
    
    print("Event: ", event)
    query = event["query"]

    print("Query: ", query)
    
    if not query:
        return {
            "statusCode": 400,
            "body": "Missing query param: 'q'"
        }

    payload = {
        "inputs": query
    }

    response = runtime.invoke_endpoint(
        EndpointName="t5-summarizer-endpoint-likhit-new",
        ContentType="application/json",
        Body=json.dumps(payload)
    )

    result = json.loads(response["Body"].read().decode())

    output_text = None
    if isinstance(result, list) and len(result) > 0:
        if "generated_text" in result[0]:
            output_text = result[0]["generated_text"]
        elif "summary_text" in result[0]:
            output_text = result[0]["summary_text"]
        elif "translation_text" in result[0]:
            output_text = result[0]["translation_text"]

    if not output_text:
        return {
            "statusCode": 500,
            "body": f"Unexpected model output format: {result}"
        }

    return {
        "statusCode": 200,
        "body": json.dumps({
            "summary": output_text
        })
    }