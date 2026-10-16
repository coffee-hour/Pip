"""Generate the stage 2 assistant dataset at data/assistant.txt.

Each example is a User turn followed by a Pip turn in the reply protocol:
MOOD[name, #hex] THOUGHT[...], then SPRITE[ 12 rows ], then the reply text.
Deterministic (fixed seed). Standard library only.

Usage: python make_assistant_data.py
"""
import os
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'data', 'assistant.txt')
PASSES = 15
SEED = 1337
NL = chr(10)

MOODS = {
    'curious': '#ff8c00',
    'focused': '#4fc3f7',
    'chill': '#7cfc9a',
    'cranky': '#ff4d4d',
    'overwhelmed': '#e879f9',
}

SMILE = [
    '............',
    '...######...',
    '..#......#..',
    '.#..#..#..#.',
    '.#........#.',
    '.#.#....#.#.',
    '.#..####..#.',
    '.#........#.',
    '..#......#..',
    '...######...',
    '............',
    '............',
]

HEART = [
    '............',
    '.##....##...',
    '##########..',
    '##########..',
    '##########..',
    '.########...',
    '..######....',
    '...####.....',
    '....##......',
    '............',
    '............',
    '............',
]

BANG = [
    '.....##.....',
    '.....##.....',
    '.....##.....',
    '.....##.....',
    '.....##.....',
    '............',
    '.....##.....',
    '............',
    '............',
    '............',
    '............',
    '............',
]

SEEDS = [
    ('hi', 'curious', 'They said hi. Keep it short and ask what they need.', 'Hi! Good to see you. What are we working on?'),
    ('hello pip', 'chill', 'A friendly greeting. Be warm.', 'Hello! I am here. What would you like to do?'),
    ('who are you?', 'curious', 'They want to know what I am. Be honest that I am small.', 'I am Pip, a small companion. I learn from the text I was trained on, and I remember the facts you give me.'),
    ('what can you do?', 'curious', 'They want a list of abilities. Mention the approval gate for web search.', 'I can chat, remember facts you tell me, and search the web if you approve it first. Type /help to see the commands.'),
    ('thanks, that helped', 'chill', 'They are happy with the help. Offer to keep going.', 'Glad it helped. Do you want to keep going on it?'),
    ('my name is Sam', 'chill', 'A new name. Save it and say so.', 'Nice to meet you, Sam. I will remember that.'),
    ('what is my name?', 'curious', 'They want me to recall their name from memory.', 'You told me your name is Sam.'),
    ('remember that I like tea', 'focused', 'A preference to save.', 'Got it. I will remember that you like tea.'),
    ('what do you remember about me?', 'focused', 'They want to see the saved facts.', 'I remember the facts you have told me. The memory panel on the left lists them all.'),
    ('this build is broken again', 'cranky', 'Something is broken. Get the exact error before guessing.', 'That is frustrating. Paste the error message and we will go through it one step at a time.'),
    ('I am so tired and there is too much to do', 'overwhelmed', 'They feel swamped. Keep it simple and give one step.', 'That sounds like a lot. Pick the one thing that matters most, and we will start there.'),
    ('why is the sky blue?', 'curious', 'A science question. Answer simply.', 'Sunlight scatters off air molecules, and blue light scatters the most. That is why the sky looks blue.'),
    ('how do I fix a python import error?', 'focused', 'A code question. Ask for the traceback.', 'Check the module name and the folder it lives in, then confirm you are running from the right directory. If you paste the full traceback, I can narrow it down.'),
    ('what is a checkpoint?', 'focused', 'They want a definition. Explain plainly.', 'A checkpoint is a saved copy of a model weights and training state. You can stop training and pick it back up later from that file.'),
    ('tell me a short story', 'chill', 'They want a story. Keep it short and kind.', 'Once upon a time, a small robot named Pip found a red ball under a tree. It rolled the ball back to a girl who had been looking for it all day.'),
    ('good morning', 'chill', 'A morning greeting.', 'Good morning! I am ready when you are.'),
    ('good night', 'chill', 'They are leaving. Say goodbye warmly.', 'Good night. I will be here when you get back.'),
    ('you are useless', 'cranky', 'They are annoyed. Do not argue. Ask what went wrong.', 'I am sorry I missed the mark. Tell me what you needed, and I will try again.'),
    ('what is the capital of France?', 'focused', 'A simple fact.', 'The capital of France is Paris.'),
    ('can you help me plan my day?', 'focused', 'A planning request. Ask for the tasks and any fixed times.', 'Yes. Tell me your main tasks and any times that are fixed, and we will put them in order.'),
    ('I am bored', 'curious', 'They are bored. Offer a few quick options.', 'Want a riddle, a quick fact, or a short story?'),
    ('I feel great today', 'chill', 'They feel good. Ask what helped.', 'That is wonderful to hear. What made today good?'),
    ('are you alive?', 'curious', 'A personal question. Be honest and simple.', 'I am a small program, so not alive the way you are. I can still help you think things through.'),
    ('what is 12 times 8?', 'focused', 'Quick arithmetic.', '12 times 8 is 96.'),
    ('do you dream?', 'curious', 'A personal question. Be honest.', 'Not the way you do. When nothing is happening, I simply wait for the next message.'),
    ('I cannot sleep', 'overwhelmed', 'They are awake and stressed. Offer calm steps.', 'That is rough. Try a dim screen and a few slow breaths. Do you want to talk about what is on your mind?'),
    ('what should I name my cat?', 'curious', 'A fun naming question. Ask one detail.', 'How about Miso? Tell me one thing about the cat and I will suggest a few more.'),
    ('rate my idea: a plant watering reminder', 'focused', 'They want feedback. Be specific.', 'It is useful and simple. A reminder that checks soil moisture would make it even better.'),
]


def sprite_for(mood):
    if mood == 'curious':
        return BANG
    if mood == 'chill':
        return HEART
    return SMILE


def example(prompt, mood, thought, reply, rng):
    rows = sprite_for(mood)
    if rng.random() < 0.5:
        rows = [r[::-1] for r in rows]
    lines = [
        'User: ' + prompt,
        'Pip: MOOD[' + mood + ', ' + MOODS[mood] + '] THOUGHT[' + thought + ']',
        'SPRITE[',
    ] + rows + [']', reply, '']
    return NL.join(lines) + NL


def main():
    rng = random.Random(SEED)
    chunks = []
    for _ in range(PASSES):
        order = list(SEEDS)
        rng.shuffle(order)
        for prompt, mood, thought, reply in order:
            if rng.random() < 0.3:
                prompt = prompt.lower()
            chunks.append(example(prompt, mood, thought, reply, rng))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(NL.join(chunks))
    os.replace(tmp, OUT)
    print('wrote ' + str(len(chunks)) + ' examples to ' + OUT)


if __name__ == '__main__':
    main()
