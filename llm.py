# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 2023
# SPDX-License-Identifier: Apache-2.0

import boto3

from langchain.llms.bedrock import Bedrock
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage

from botocore.config import Config

from .env import get_bedrock_region

available_models = [
    "amazon.titan-tg1-large",
    "amazon.nova-premier-v1:0",
    "amazon.nova-pro-v1:0"
]

# the current model used for text generation
text_model_id = "amazon.nova-pro-v1:0"
text_v2_model_id = "amazon.nova-premier-v1:0"
embedding_model_id = "amazon.titan-embed-text-v2:0"

model_kwargs = {
    "amazon.titan-tg1-large": {
        "temperature": 0, 
        "maxTokenCount": 4096,
    },
    "amazon.nova-pro-v1:0": {
        "messages": [
            {"role": "user", "content": ""}  # This will be replaced with actual content
        ],
        "temperature": 0.1,
        "topP": 0.9,
        "maxTokens": 4096,
    },
    "amazon.nova-premier-v1:0": {
        "messages": [
            {"role": "user", "content": ""}  # This will be replaced with actual content
        ],
        "temperature": 0.1,
        "topP": 0.9,
        "maxTokens": 4096,
    },
}

prompt_template_prefix = {
    "amazon.nova-pro-v1": "",
    "amazon.nova-premier-v1": ""
}

prompt_template_postfix = {
    "amazon.nova-pro-v1": "",
    "amazon.nova-premier-v1": ""
}

def get_template_proc(model_id):
    def template_proc(template):
        return f"{prompt_template_prefix[model_id]}{template}{prompt_template_postfix[model_id]}"
    return template_proc

prompt_template_procs = {
    "amazon.nova-pro-v1": get_template_proc("amazon.nova-pro-v1"),
    "amazon.nova-premier-v1": get_template_proc("amazon.nova-premier-v1")
}

bedrock = boto3.client('bedrock', get_bedrock_region(), config=Config(
    retries = {
        'max_attempts': 10,
        'mode': 'standard'
    }
))
bedrock_runtime = boto3.client('bedrock-runtime', get_bedrock_region(), config=Config(
    retries = {
        'max_attempts': 10,
        'mode': 'standard'
    }
))
response = bedrock.list_foundation_models()
print(response.get('modelSummaries')) 

def get_bedrock_embedding():
    embeddings = BedrockEmbeddings(
        model_id=embedding_model_id,
        client=bedrock_runtime
    )
    return embeddings

def get_bedrock_text(model_id=None, temperature=0.7):
    """
    Creates a ChatBedrockConverse instance for chat-based interactions with Bedrock models.
    
    Args:
        model_id (str, optional): The model ID to use. Defaults to text_model_id if None.
        temperature (float, optional): The temperature setting. Defaults to 0.7.
        
    Returns:
        ChatBedrockConverse: A configured chat model instance.
    """
    if model_id is None:
        model_id = text_v2_model_id
        
    llm = ChatBedrockConverse(
        model_id=model_id,
        client=bedrock_runtime,
        temperature=temperature
    )
    
    return llm

def get_bedrock_text(model_id=None, temperature=0.7):
    """
    Creates a ChatBedrockConverse instance for chat-based interactions with Bedrock models.
    
    Args:
        model_id (str, optional): The model ID to use. Defaults to text_model_id if None.
        temperature (float, optional): The temperature setting. Defaults to 0.7.
        
    Returns:
        ChatBedrockConverse: A configured chat model instance.
    """
    if model_id is None:
        model_id = text_model_id
        
    llm = ChatBedrockConverse(
        model_id=model_id,
        client=bedrock_runtime,
        temperature=temperature
    )
    
    return llm


def get_processed_prompt_template(template):
    if text_model_id in prompt_template_procs:
        return prompt_template_procs[text_model_id](template)
    else:
        return template

def get_prefix_prompt_template(template):
    if text_model_id in prompt_template_prefix:
        return prompt_template_prefix[text_model_id] + template
    else:
        return template

def get_postfix_prompt_template(template):
    if text_model_id in prompt_template_postfix:
        return template + prompt_template_postfix[text_model_id]
    else:
        return template
