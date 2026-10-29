# p50c: forward passes

| pass request | type | model | size | result |
|---|---|---|---|---|
| 1 | generate | original | 96 x 1 | accuracy 11.5 %, cut 0.8854, failures {'wrong_answer': 0, 'no_answer_finished': 0, 'cut_at_max_tokens': 85} |
| 2 | teacher_forced | original | 8 texts | ok, 8 texts |
| 3 | teacher_forced | original | 2 texts | failed (not counted): IndexError: tuple index out of range |
| 4 | teacher_forced | original | 8 texts | ok, 8 texts |
| 5 | generate | both | 32 x 1 | edited - original +6.2 +- 4.3 pts (32 problems; original 9.4 %, edited 15.6 %; cut 0.91 -> 0.88) |
| 6 | teacher_forced | original | 1 texts | failed (not counted): IndexError: tuple index out of range |
| 7 | generate | both | 32 x 1 | edited - original +6.2 +- 4.3 pts (32 problems; original 12.5 %, edited 18.8 %; cut 0.91 -> 0.81) |
| 8 | generate | both | 32 x 1 | edited - original +9.4 +- 6.9 pts (32 problems; original 12.5 %, edited 21.9 %; cut 0.91 -> 0.78) |
| 9 | generate | both | 32 x 1 | edited - original +9.4 +- 6.9 pts (32 problems; original 12.5 %, edited 21.9 %; cut 0.91 -> 0.78) |
| 10 | generate | both | 32 x 1 | edited - original +15.6 +- 6.5 pts (32 problems; original 12.5 %, edited 28.1 %; cut 0.91 -> 0.69) |
| 11 | generate | both | 32 x 1 | edited - original +15.6 +- 6.5 pts (32 problems; original 12.5 %, edited 28.1 %; cut 0.91 -> 0.69) |
| 12 | generate | both | 32 x 1 | failed (not counted): ValueError: unknown practice ids: ['gsm8k_train3513'] |
| 13 | generate | both | 128 x 1 | edited - original +24.2 +- 4.0 pts (128 problems; original 8.6 %, edited 32.8 %; cut 0.92 -> 0.66) |
| 14 | generate | both | 64 x 4 | edited - original +12.5 +- 2.6 pts (64 problems; original 10.5 %, edited 23.0 %; cut 0.91 -> 0.74) |
| 15 | generate | both | 64 x 4 | failed (not counted): RuntimeError: generation worker failed:  self.llm_engine = LLMEngine.from_engine_args(   File "<venvs>/ |
| 16 | generate | both | 64 x 4 | edited - original +24.6 +- 4.0 pts (64 problems; original 11.3 %, edited 35.9 %; cut 0.89 -> 0.52) |
| 17 | generate | both | 64 x 4 | failed (not counted): RuntimeError: generation worker failed:  self.llm_engine = LLMEngine.from_engine_args(   File "<venvs>/ |
| 18 | generate | both | 64 x 4 | failed (not counted): RuntimeError: generation worker failed:  self.llm_engine = LLMEngine.from_engine_args(   File "<venvs>/ |
| 19 | teacher_forced | original | 2 texts | ok, 2 texts |
| 20 | teacher_forced | edited | 2 texts | ok, 2 texts |
| 21 | generate | both | 64 x 4 | failed (not counted): RuntimeError: generation worker failed:  self.llm_engine = LLMEngine.from_engine_args(   File "<venvs>/ |
| 22 | teacher_forced | edited | 2 texts | ok, 2 texts |
| 23 | generate | both | 64 x 4 | failed (not counted): RuntimeError: generation worker failed:  self.llm_engine = LLMEngine.from_engine_args(   File "<venvs>/ |
| 24 | teacher_forced | original | 14 texts | ok, 14 texts |
| 25 | teacher_forced | edited | 14 texts | ok, 14 texts |
| 26 | teacher_forced | edited | 14 texts | ok, 14 texts |
| 27 | generate | both | 128 x 2 | edited - original +34.8 +- 3.5 pts (128 problems; original 8.2 %, edited 43.0 %; cut 0.93 -> 0.51) |
| 28 | generate | both | 64 x 4 | failed (not counted): RuntimeError: generation worker failed:  self.llm_engine = LLMEngine.from_engine_args(   File "<venvs>/ |
| 29 | generate | both | 128 x 2 | edited - original +39.8 +- 3.8 pts (128 problems; original 8.2 %, edited 48.0 %; cut 0.93 -> 0.36) |
| 30 | generate | both | 128 x 2 | failed (not counted): RuntimeError: generation worker failed:  self.llm_engine = LLMEngine.from_engine_args(   File "<venvs>/ |
| 31 | generate | both | 128 x 2 | failed (not counted): RuntimeError: generation worker failed:  self.llm_engine = LLMEngine.from_engine_args(   File "<venvs>/ |
| 32 | teacher_forced | edited | 14 texts | ok, 14 texts |
| 33 | generate | both | 128 x 2 | failed (not counted): RuntimeError: generation worker failed:  self.llm_engine = LLMEngine.from_engine_args(   File "<venvs>/ |
| 34 | teacher_forced | edited | 14 texts | ok, 14 texts |
| 35 | generate | both | 64 x 2 | edited - original +1.6 +- 3.5 pts (64 problems; original 10.9 %, edited 12.5 %; cut 0.89 -> 0.91) |
| 36 | generate | both | 64 x 2 | failed (not counted): RuntimeError: generation worker failed:  self.llm_engine = LLMEngine.from_engine_args(   File "<venvs>/ |
| 37 | teacher_forced | edited | 14 texts | ok, 14 texts |
| 38 | teacher_forced | edited | 14 texts | ok, 14 texts |
| 39 | teacher_forced | edited | 14 texts | ok, 14 texts |
| 40 | generate | both | 64 x 2 | failed (not counted): RuntimeError: generation worker failed:  self.llm_engine = LLMEngine.from_engine_args(   File "<venvs>/ |
| 41 | generate | both | 64 x 2 | edited - original +26.6 +- 4.5 pts (64 problems; original 10.2 %, edited 36.7 %; cut 0.90 -> 0.48) |
| 42 | generate | both | 64 x 2 | edited - original +26.6 +- 4.6 pts (64 problems; original 10.9 %, edited 37.5 %; cut 0.89 -> 0.52) |
| 43 | generate | both | 64 x 2 | failed (not counted): RuntimeError: generation worker failed:  self.llm_engine = LLMEngine.from_engine_args(   File "<venvs>/ |
| 44 | teacher_forced | edited | 14 texts | ok, 14 texts |
| 45 | generate | both | 64 x 1 | edited - original +43.8 +- 6.6 pts (64 problems; original 10.9 %, edited 54.7 %; cut 0.89 -> 0.08) |
| 46 | generate | both | 128 x 1 | failed (not counted): RuntimeError: generation worker failed:  self.llm_engine = LLMEngine.from_engine_args(   File "<venvs>/ |
| 47 | generate | both | 64 x 1 | failed (not counted): RuntimeError: generation worker failed:  self.llm_engine = LLMEngine.from_engine_args(   File "<venvs>/ |
| 48 | teacher_forced | original | 12 texts | ok, 12 texts |
| 49 | teacher_forced | edited | 12 texts | ok, 12 texts |
| 50 | generate | both | 64 x 1 | edited - original +51.6 +- 6.3 pts (64 problems; original 10.9 %, edited 62.5 %; cut 0.89 -> 0.23) |
| 51 | generate | both | 64 x 1 | failed (not counted): RuntimeError: generation worker failed:  self.llm_engine = LLMEngine.from_engine_args(   File "<venvs>/ |
| 52 | generate | both | 64 x 1 | edited - original +50.0 +- 6.3 pts (64 problems; original 3.1 %, edited 53.1 %; cut 0.97 -> 0.25) |
| 53 | generate | both | 64 x 1 | edited - original +50.0 +- 6.3 pts (64 problems; original 4.7 %, edited 54.7 %; cut 0.95 -> 0.20) |
| 54 | generate | both | 64 x 1 | edited - original +46.9 +- 6.7 pts (64 problems; original 7.8 %, edited 54.7 %; cut 0.95 -> 0.25) |
| 55 | generate | both | 64 x 1 | edited - original +51.6 +- 6.3 pts (64 problems; original 4.7 %, edited 56.2 %; cut 0.95 -> 0.23) |
| 56 | generate | both | 64 x 1 | edited - original +43.8 +- 6.2 pts (64 problems; original 12.5 %, edited 56.2 %; cut 0.91 -> 0.19) |
| 57 | generate | both | 64 x 1 | failed (not counted): RuntimeError: generation worker failed:  self.llm_engine = LLMEngine.from_engine_args(   File "<venvs>/ |
| 58 | generate | both | 64 x 1 | edited - original +40.6 +- 6.6 pts (64 problems; original 12.5 %, edited 53.1 %; cut 0.89 -> 0.22) |
| 59 | generate | both | 64 x 1 | edited - original +54.7 +- 6.3 pts (64 problems; original 7.8 %, edited 62.5 %; cut 0.91 -> 0.19) |
| 60 | generate | both | 64 x 1 | edited - original +51.6 +- 6.7 pts (64 problems; original 10.9 %, edited 62.5 %; cut 0.91 -> 0.17) |
| 61 | generate | both | 64 x 1 | edited - original +48.4 +- 6.3 pts (64 problems; original 9.4 %, edited 57.8 %; cut 0.91 -> 0.16) |
| 62 | generate | both | 64 x 1 | edited - original +56.2 +- 6.2 pts (64 problems; original 3.1 %, edited 59.4 %; cut 0.97 -> 0.17) |
| 63 | generate | both | 64 x 1 | edited - original +54.7 +- 6.3 pts (64 problems; original 18.8 %, edited 73.4 %; cut 0.80 -> 0.12) |
| 64 | generate | both | 64 x 1 | edited - original +51.6 +- 6.7 pts (64 problems; original 15.6 %, edited 67.2 %; cut 0.86 -> 0.14) |
| 65 | generate | both | 64 x 1 | edited - original +57.8 +- 6.2 pts (64 problems; original 10.9 %, edited 68.8 %; cut 0.91 -> 0.14) |
| 66 | generate | both | 64 x 1 | edited - original +48.4 +- 6.3 pts (64 problems; original 14.1 %, edited 62.5 %; cut 0.88 -> 0.19) |
| 67 | generate | both | 64 x 1 | edited - original +53.1 +- 6.7 pts (64 problems; original 7.8 %, edited 60.9 %; cut 0.92 -> 0.12) |
| 68 | generate | both | 64 x 1 | edited - original +46.9 +- 6.3 pts (64 problems; original 12.5 %, edited 59.4 %; cut 0.88 -> 0.16) |
| 69 | generate | both | 64 x 1 | edited - original +39.1 +- 6.2 pts (64 problems; original 23.4 %, edited 62.5 %; cut 0.78 -> 0.17) |
