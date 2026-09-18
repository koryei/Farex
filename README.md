# My research project
What is this? ->
- I study what happens when a multi-agent system sends tasks to the wrong agent.
- I use 4 agent types: coding, math, extraction, evidence-qa.
- I test 5 conditions: perfect routing, random routing, wrong routing, local router, local + escalation.

How to run (simple steps) ->
1. Install Ollama on your Mac / server.
2. Pull 2 local models (example: llama3.1, mistral-nemo).
3. Put your tasks in /tasks/ .
4. Run evaluate.py or your script. It writes to /logs/.

What I measure ->
- Routing accuracy (was the right agent chosen?)
- Task success (did the answer match?)
- Tokens used
- Time (latency)

Files ->
- /tasks/ : the 10-50 test questions
- /agents/ : the 4 agent prompts
- /logs/ : every run saves a JSON file
- /evaluate/ : my counting script

License ->
MIT License - Code assist and draft structure used AI tools; experiment design, task selection, evaluation logic, and interpretation by Aarav Banshiwala.

Contact ->
If you want to learn or give feedback: aaravbanshiwala@gmail.com
