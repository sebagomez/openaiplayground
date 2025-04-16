Create Python environment from https://linuxize.com/post/how-to-create-python-virtual-environments-on-ubuntu-18-04/


```sh
apt install python3.10-venv
python3 -m venv .venv
source .venv/bin/activate
```

Install requirements

```sh
pip install -r requirements.txt
```

Using the following question

```python
question = "Who won the first GP of the 2025 Formula 1 season?"
```

ChatGPT answers  
> I'm sorry, but as an AI, I do not have real-time information. Please check the latest updates or news for the winner of the first GP of the 2025 Formula 1 season.

Since it's April 2025, the current version of ChatGPT does not know the answer.  
After feeding the document [Formula 1 - 2025 Season_ocr.pdf](./data/Formula%201%20-%202025%20Season_ocr.pdf) which it's a print from the Wikipedia, ChatGPT answers the following to the same question.
> Lando Norris won the Australian Grand Prix, the first GP of the 2025 Formula 1 season, with Max Verstappen finishing behind him. Oscar Piastri, Lando Norris' teammate, took pole position for the Chinese Grand Prix and won the main race. Oscar Piastri also won the Bahrain Grand Prix, the third GP of the 2025 season.

This excercise came from the LinkedIn Learning course [Chat with your data using ChatGPT](https://www.linkedin.com/learning/chat-with-your-data-using-chatgpt)
