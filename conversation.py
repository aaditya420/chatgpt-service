import os
import openai
import dotenv
import tiktoken

dotenv.load_dotenv("dev.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

class Conversation:

    def __init__(self, model_name, token_limit, system_message, pre_text = None, encoding_name = "gpt2") -> None:
        
        self.conversation = [
            {
                "role": "system",
                "content": system_message
            }
        ]

        if pre_text is not None:
            self.conversation.extend(pre_text)
        
        self.model_name = model_name
        self.token_limit = token_limit
        self.encoding_name = encoding_name
        self.system_message = system_message

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(self.encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    
    def reset_conversation(self, system_message=None):
        if system_message is None:
            system_message = self.system_message
        self.conversation = [
            {
                "role": "system",
                "content": system_message
            }
        ]

    def chat(self, role, content):

        conversation_string = " ".join([convo["content"] for convo in self.conversation])
        conversation_tokens = self.num_tokens_from_string(conversation_string)
        if conversation_tokens > self.token_limit:
            print("Too much text!")
            return None

        # Check content tokens
        tokens = self.num_tokens_from_string(content)
        if (tokens + conversation_tokens) > self.token_limit:
            print("Too much text!")
            return None

        self.conversation.append({
            "role": role,
            "content": content
        })

        res = openai.ChatCompletion.create(
            model=self.model_name,
            messages=self.conversation
        )

        self.conversation.append(res['choices'][0]['message'])

        return res['choices'][0]['message']['content']


if __name__ == "__main__":

    new_convo = Conversation(
        model_name="gpt-4",
        token_limit=4096,
        system_message="You are a very helpful bot who is a proficient engineer and developer by progression, you know how to follow best coding practices and have a lot of industry standard experience"
    )
    while True:
        user_input = input("User: ")
        print("ChatGPT: " + new_convo.chat(
            role="user",
            content=user_input
        ))