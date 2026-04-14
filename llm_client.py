"""
Gemini client wrapper for DocuBot Article Analyser.

Provides a single generate() method used for both query expansion
and synthesis prompts.
"""

import os
import google.generativeai as genai

GEMINI_MODEL_NAME = "gemini-2.5-flash"


class GeminiClient:
    """
    Thin wrapper around the Gemini generative model.

    Usage:
        client = GeminiClient()
        text = client.generate(prompt)
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY environment variable. "
                "Set it in your shell or .env file to enable LLM features."
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    def generate(self, prompt: str) -> str:
        """Send a prompt and return the response text."""
        response = self.model.generate_content(prompt)
        return (response.text or "").strip()
