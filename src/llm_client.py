import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        chat_args: dict = {}
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.chat_args = chat_args

    def query_chat_completions(self, user_prompt, system_prompt=None, extra_args=None):
        args = {**self.chat_args, **(extra_args or {})}
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            **args
        )
        response_text = response.choices[0].message.content.strip()
        reasoning_content = ""
        if hasattr(response.choices[0].message, "reasoning_content"):
            reasoning_content = response.choices[0].message.reasoning_content.strip()

        return response_text, reasoning_content

    def query_response(self, user_prompt, system_prompt=None):
        response = self.client.responses.create(
            model=self.model_name,
            input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            **self.chat_args
        )
        return response.output_text, ""

    def query(self, user_prompt, system_prompt=None, return_reasoning_content=False, extra_args=None):
        logger.info(f'Querying {self.model_name}')
        if self.model_name.startswith('gpt-5.'):
            response_text, reasoning_content = self.query_response(user_prompt, system_prompt)
        else:
            response_text, reasoning_content = self.query_chat_completions(user_prompt, system_prompt, extra_args)
        if return_reasoning_content:
            return response_text, reasoning_content
        return response_text
