# ChatGPT API Wrapper

A Python wrapper for the ChatGPT API, providing an easy-to-use interface for interacting with the powerful OpenAI ChatGPT model, while also supporting conversation history and token count management.

## Features

- Simple and efficient communication with the ChatGPT API
- Conversation history support for multi-turn conversations
- Token counts enablement for efficient token management
- Follows best coding practices and industry-standard design patterns
- Comprehensive documentation for seamless onboarding and usage

## Installation

To install the chatgpt_api_wrapper package, run:

```
pip install chatgpt_api_wrapper
```

## Usage

Here's a quick example of how to use the `chatgpt_api_wrapper` to generate a response from the ChatGPT model:

```python
from chatgpt_api_wrapper import ChatGPT

# Initialize ChatGPT wrapper with your API key
api_key = "your_openai_api_key"
chatgpt = ChatGPT(api_key)

# Set up inputs and settings you want to use
messages = [
    {"role": "system", "content": "Welcome to ChatGPT."},
    {"role": "user", "content": "Tell me a joke."},
]
max_tokens = 50
temperature = 0.8

# Generate response using the wrapper
response = chatgpt.generate_response(messages, max_tokens=max_tokens, temperature=temperature)

print(response)
```

For more detailed usage and API documentation, please refer to the [documentation](link_to_doc).

## TODO

[ ] Convert to pip package

[ ] Include tests

[ ] Add DB Support for chat history

## Contributing

Contributions are welcome! Please follow these steps if you'd like to contribute:

1. Fork the repository.
2. Create a branch named `feature/your-feature` or `bugfix/your-bugfix`.
3. Commit your changes to the branch.
4. Create a pull request describing the changes made and reference any related issues.
5. Make sure to follow best coding practices and write clear, concise code.

## Support

If you encounter any issues or have questions, please open an issue or check the [documentation](link_to_doc) for further information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.