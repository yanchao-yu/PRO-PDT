# Edinburgh Napier PDT Chatbot

A reverse-roleplay Personal Development Tutor (PDT) chatbot prototype for Edinburgh Napier University, built for the **Machine Learning for Conversational AI** group project.

This project uses:
- **DeepSeek** for dialogue generation / reasoning
- **Furhat** for embodied interaction
- structured **YAML knowledge files** for retrieval, routing, safety, and evaluation

The goal is to create a chatbot that can:
- answer common PDT-style student questions
- stay within the real PDT role
- signpost students to the correct support services
- handle safety and escalation appropriately
- retrieve grounded information about courses and modules when needed

The PDT role at Napier is a supportive, proactive academic/pastoral contact who helps students reflect on academic progress, personal development, and access to appropriate support. PDTs should signpost students to relevant services, aim to respond within three working days, and should **not** claim to change grades or replace specialist services. 

## Repository purpose

This repository is intended to support:
- dialogue system prototyping
- data collection and structured roleplay design
- retrieval-based question answering
- Furhat dialogue testing
- evaluation of safety, role consistency, and signposting quality

The coursework requires a demonstrative system, code repository, and a comprehensive README, so this repo is organised to support both the system itself and the written/reporting side of the project.

## Core data files

### Knowledge and lookup
- **`all_courses.yaml`**  
  Full structured course catalogue. Use for general course lookup by course title, award, academic level, and study mode. 

- **`pdt_course_lookup.yaml`**  
  Smaller trimmed course/module lookup file for demos and lightweight retrieval. This is a starter subset, not a full institutional record. {index=3}

- **`pdt-data.yaml`**  
  Main **module datastore**. Use this when a student asks about a module code, module title, SCQF level, credit value, prerequisites, module leader, description, or learning objectives. 

- **`napier_pdt_support.yaml`**  
  Main Napier support knowledge base. Includes PDT guidance, counselling and mental wellbeing, Togetherall, emergency/out-of-hours support, and exam results guidance. 

- **`pdt_services.yaml`**  
  Simplified service directory used for **routing and signposting**. Includes internal support services and urgent/external support options.

### Behaviour, safety, and routing
- **`pdt_persona_policy.yaml`**  
  Defines the PDT persona, tone, role boundaries, allowed behaviours, forbidden behaviours, confidentiality rules, and response framework.

- **`pdt_intents.yaml`**  
  Defines student intents, entities, sample utterances, and expected system actions. Use this for intent classification and dialogue routing. 

- **`pdt_safety_escalation.yaml`**  
  Safety and escalation layer. Defines risk levels, crisis keywords, confidentiality limits, safe response rules, and recommended routing for urgent situations.

### Dialogues and evaluation
- **`pdt_dialogues.yaml`**  
  Starter roleplay dataset containing PDT/student examples for prompting, expansion, and testing. Includes both standard and reverse-roleplay patterns.

- **`pdt_eval_set.yaml`**  
  Evaluation scenarios for testing role consistency, helpfulness, safety, empathy, signposting accuracy, and boundary adherence.

## Suggested system architecture

### 1. Input layer
Furhat receives the student utterance.

### 2. Intent and risk detection
Use:
- `pdt_intents.yaml` for intent classification
- `pdt_safety_escalation.yaml` for urgency / crisis detection 

### 3. Retrieval layer
Route the query to the right datastore:
- **course questions** → `all_courses.yaml` / `pdt_course_lookup.yaml` 
- **module questions** → `pdt-data.yaml` 
- **support/service/signposting questions** → `pdt_services.yaml` / `napier_pdt_support.yaml` 

### 4. Response control
Constrain generation using:
- `pdt_persona_policy.yaml`
- `pdt_safety_escalation.yaml` 

### 5. Dialogue generation
DeepSeek generates the reply using:
- retrieved context
- role/policy instructions
- conversation history

### 6. Evaluation
Test outputs against:
- `pdt_eval_set.yaml`
- selected examples from `pdt_dialogues.yaml` 

## Recommended use of the data

### Use `pdt-data.yaml` as a module lookup store
This file is especially useful for:
- exact module code lookup
- fuzzy module title matching
- retrieving module descriptions
- answering broad “what is this module about?” questions
- retrieving prerequisites and learning objectives when grounded in the file

### Use `pdt_services.yaml` for signposting
This should be the main routing directory when students ask:
- who should I contact
- where can I get support
- what should I do next
- what support exists for stress, anxiety, study skills, results problems, or urgent help 

### Use `pdt_persona_policy.yaml` and `pdt_safety_escalation.yaml` as hard constraints
These files stop the chatbot from:
- pretending to access personal records
- changing grades
- diagnosing mental health conditions
- acting as a crisis counsellor
- overpromising confidentiality in risk situations 

## Current limitations

- The course data is useful for lookup, but not a full official programme-structure map. `pdt_course_lookup.yaml` explicitly notes that it is a trimmed starter set.
- The module file is broad and includes many university subjects, so it should be treated as a lookup datastore rather than the core PDT knowledge base. 
- The dialogue set is a strong prototype seed, but should still be expanded further for broader coverage and more robust evaluation. 

## Recommended next steps

- expand dialogue coverage further
- align all routing/service IDs across files
- add a module search utility and service search utility in code
- export training/evaluation data into machine-friendly formats
- connect Furhat input/output to the retrieval + policy + generation pipeline

## Team summary

This repo currently gives us:
- a structured **course lookup**
- a structured **module lookup**
- a structured **support/service directory**
- a defined **PDT persona and boundary policy**
- a defined **safety and escalation layer**
- a starter **dialogue dataset**
- a starter **evaluation set**

Together, these files are enough to support a strong prototype/demo of a PDT chatbot with grounded retrieval, safe role behaviour, and Furhat-compatible conversational design.

## Vision Integration (Face Recognition)

This project now includes an optional **face recognition module** via the `recognition` submodule. The vision system automatically:
- Detects when users enter/leave the robot's field of view
- Identifies users by face
- Determines user role (student/staff) 
- Adapts dialogue and voice based on user role

### How Vision Integration Works

The vision system sends events to the chatbot when:
1. **User Changed**: A user is detected with a name and role
   - **Student role**: Switches to tutor dialogue mode with female voice
   - **Staff role**: Switches to staff assistance mode with male voice
   - **Unknown user**: Prompts the robot to ask for their name

2. **User Left**: A user leaves the robot's field of view
   - Robot says goodbye with the user's name

### Running with Vision

```bash
# Option 1: Using the vision_example.py helper
python vision_example.py

# Option 2: Using furhat_bridge.py directly with --enable_vision flag
python furhat_bridge.py --enable_vision --host 127.0.0.1
```

### Prerequisites for Vision

Make sure the face-recognition-on-a-robot submodule is initialized:
```bash
git submodule update --init --recursive
cd recognition
# Follow that project's setup instructions
```

## To run this program
- Use **Python 3.10** for the full Rasa + Furhat setup.
- `pip install -r requirements.txt` installs the full stack on Python 3.10.
- On Python 3.11+ it will skip `rasa` automatically so the rest of the dependencies can still be installed.
- Furhat SDK version 2.9.2 using the Remote API.
- If you want to train and run the Rasa assistant, create and activate a Python 3.10 virtual environment first.
- rasa train
- rasa run actions
- rasa run --enable-api --cors "*"
- python furhat_bridge.py (can change host --host & authorization key --auth_key if required)
