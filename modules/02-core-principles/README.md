# Module 2: Core principles, terms, technologies (Week 2, ~10 hours)

## Hours 1-3: the five-layer map
Read the five-layer breakdown in `CURRICULUM.md`. Then take any vendor architecture diagram (an "AI platform" slide from a MarTech vendor works well) and label every box with its layer. Boxes that fit no layer are usually marketing.

## Hours 4-6: `annotated_call.py`
```bash
python modules/02-core-principles/annotated_call.py
```
With no API key it prints the request it would send. With a key it makes the call. Every line is commented with the concept it demonstrates. Tasks:
- Change `temperature` to 1.0 and run three times. Then back to 0 and run three times. Note what varies.
- Remove the `tools` block and ask the same question. Note what the model does when it cannot call the tool.
- Add `cache_control` on the system prompt and check `usage.cache_read_input_tokens` on the second call.

## Hours 7-8: glossary self-test
Fill in `glossary_selftest.md` from memory. Then check against `GLOSSARY.md`. Anything you got wrong, write a one-line example from your own domain.

## Hours 9-10: prompt injection in your domain
Read the OWASP LLM Top 10 entry for prompt injection. Write down two ways a lead-form submission or a scraped competitor page could inject instructions into the Module 4 agent, and what in `answer.py` or `tools.py` would stop it. (Hint: the agent only ever sees tool output as data, and numbers come from code; but document chunks are text.)

## Done when
Glossary self-test ≥ 90%, the three `annotated_call.py` experiments recorded, and the injection note written.
