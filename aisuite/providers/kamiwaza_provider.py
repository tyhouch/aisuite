import os
from typing import Optional
import openai
from kamiwaza_client import KamiwazaClient

from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse


class KamiwazaProvider(Provider):
    """
    Provider implementation for Kamiwaza.
    
    This provider allows using Kamiwaza's models through aisuite's unified interface.
    It automatically handles deployment discovery and port management.
    
    Configuration options:
        - base_url: The base URL for the Kamiwaza API (default: http://localhost:7777/api/)
        - timeout: Request timeout in seconds (default: 30)
    """

    def __init__(self, **config):
        """
        Initialize the Kamiwaza provider with given configuration.
        
        Args:
            **config: Configuration options including optional base_url and timeout
        """
        self.base_url = config.get("base_url", "http://localhost:7777/api/")
        self.timeout = config.get("timeout", 30)
        
        try:
            self.client = KamiwazaClient(self.base_url)
        except Exception as e:
            raise LLMError(f"Failed to initialize Kamiwaza client: {str(e)}")

    def _find_valid_deployment(self, model_name: str) -> Optional[dict]:
        """
        Find a valid deployment for the given model name.
        
        Args:
            model_name: Name of the model to find a deployment for
            
        Returns:
            The first valid deployment found for the model, or None if no valid deployment exists
        """
        try:
            deployments = self.client.serving.list_deployments()
            # Find first deployment that matches model name and is deployed with instances
            valid_deployment = next(
                (d for d in deployments 
                 if d.status == 'DEPLOYED' and 
                 d.instances and 
                 d.m_name == model_name),
                None
            )
            return valid_deployment
        except Exception as e:
            raise LLMError(f"Failed to list deployments: {str(e)}")

    def chat_completions_create(self, model: str, messages: list, **kwargs):
        """
        Create a chat completion using a Kamiwaza model.
        
        Args:
            model: Name of the model to use
            messages: List of conversation messages
            **kwargs: Additional parameters to pass to the completion API
            
        Returns:
            ChatCompletionResponse containing the model's response
            
        Raises:
            LLMError: If no valid deployment is found or if the completion request fails
        """
        # Find valid deployment for the model
        deployment = self._find_valid_deployment(model)
        if not deployment:
            raise LLMError(
                f"No valid deployment found for model '{model}'. "
                "Ensure the model is deployed and has running instances."
            )

        # Create OpenAI client configured for the deployment
        try:
            openai_client = openai.OpenAI(
                api_key="local",  # Kamiwaza uses "local" as API key
                base_url=f"http://localhost:{deployment.lb_port}/v1",
                timeout=self.timeout
            )
            
            # Make the completion request
            # Note: Kamiwaza expects "local-model" as the model name in the actual request
            response = openai_client.chat.completions.create(
                model="local-model",
                messages=messages,
                **kwargs
            )
            
            return response
            
        except Exception as e:
            raise LLMError(
                f"Chat completion failed for model '{model}' "
                f"on port {deployment.lb_port}: {str(e)}"
            )

    def __repr__(self) -> str:
        return f"KamiwazaProvider(base_url={self.base_url})"