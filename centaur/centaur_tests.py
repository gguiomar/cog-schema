#%%

!pip install unsloth "xformers==0.0.28.post2"

#%%

from unsloth import FastLanguageModel
import transformers

model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = "marcelbinz/Llama-3.1-Centaur-8B-adapter",
  max_seq_length = 32768,
  dtype = None,
  load_in_4bit = True,
)

#%%

FastLanguageModel.for_inference(model)

pipe = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            pad_token_id=0,
            do_sample=True,
            temperature=1.0,
            max_new_tokens=1,
)

#%%

prompt = "You will be presented with triplets of objects, which will be assigned to the keys H, Y, and E.\n" \
  "In each trial, please indicate which object you think is the odd one out by pressing the corresponding key.\n" \
  "In other words, please choose the object that is the least similar to the other two.\n\n" \
  "H: plant, Y: chainsaw, and E: periscope. You press <<H>>.\n" \
  "H: tostada, Y: leaf, and E: sail. You press <<H>>.\n" \
  "H: clock, Y: crystal, and E: grate. You press <<Y>>.\n" \
  "H: barbed wire, Y: kale, and E: sweater. You press <<E>>.\n" \
  "H: car, Y: dog, and E: elephant. You press <<"

choice = pipe(prompt)[0]['generated_text'][len(prompt):]
print(choice)

#%%

prompt  = "You will be asked to repeatedly choose between four different options labeled L, G, O, and U.\n" \
"You select an option by pressing the corresponding key on your keyboard.\n"\
"Each time you select an option, you will get a different number of points.\n"\
"Your goal is to win as many points as possible.\n"\
"You press <<"

choice = pipe(prompt)[0]['generated_text'][len(prompt):]
print(choice)

#%%

prompt = "Throughout the task, you will be presented with balloons, one at a time."\
"In each step, you can choose to pump up the balloon by pressing H and you will accu- mulate 1 point for each pump."\
"At any point, you can stop pumping up the balloon by pressing W and you will col- lect your accumulated points."\
"You will repeat this procedure on multiple different balloons."\
"It is your choice to determine how much to pump up the balloon, but be aware that at some point the balloon will explode."\
"If the balloon explodes before you collect your accumulated points, then you move on to the next balloon and the points are lost."\
"Balloon 1: "\
"You press <<"


#%%


main_prompt = "You will repeatedly observe sequences of six letters.\n"\
"You have to remember these letters before they disappear.\n"\
"Afterward, you will be prompted with one letter. You have to answer whether the letter was part of the six previous letters.\n"\
"If you think it was, you have to press 1. If you think it was not, press 0.\n"\
"You are shown the letters [’C’, ’I’, ’N’, ’K’, ’W’, ’Z’].
letters = ['C', 'I', 'N', 'K', 'W', 'Z']
letter = random.choice(letters)
prompt = f"You see the letter {letter}. You press <<"


#%%

main_prompt = "You will repeatedly observe sequences of six letters.\n"\
"You have to remember these letters before they disappear.\n"\
"Afterward, you will be prompted with one letter. You have to answer whether the letter was part of the six previous letters.\n"\
"If you think it was, you have to press 1. If you think it was not, press 0.\n"\
"You are shown the letters [’C’, ’I’, ’N’, ’K’, ’W’, ’Z’]"
#choice = pipe(main_prompt)[0]['generated_text'][len(prompt):]
for i in range(1):
    letter = random.choice(letters)
    prompt = f"{main_prompt}. You see the letter {letter}. You press <<"
    print(prompt)
    choice = pipe(prompt)[0]['generated_text'][len(prompt):]
    print(prompt, choice, ">>")