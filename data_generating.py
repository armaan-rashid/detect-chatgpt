from revChatGPT.V1 import Chatbot
from argparse import ArgumentParser
from sys import stdout
import asyncio

argparser = ArgumentParser(prog='ChatGPT Scraper', description='Generate tokens and responses from ChatGPT using unofficial API.')
argparser.add_argument('email', help='openAI login')
argparser.add_argument('password', help='openAI login')
argparser.add_argument('--plus', action='store_true', help='specify option if you have ChatGPT Plus', default=False)
files_or_prompts = argparser.add_mutually_exclusive_group(required=True)
files_or_prompts.add_argument('-f', '--files', nargs='*', help='Files where each line is a set of tokens for prompting ChatGPT. \
                                                         Expects .txt files with a prompt on each line of the file. \
                                                         Responses will be placed in new files with _responses\
                                                         at the end of the filename. -p and cannot be specified.', default=[])
files_or_prompts.add_argument('-p', '--prompts', nargs='*', help='Individual prompts for ChatGPT. Responses are written in \
                                                           in destination file if specified, or printed. -f cannot be specified.', default=[])
argparser.add_argument('-r', '--retain', action='store_true', help='If this option is specified, write both prompt \
                                                                    and response together, separated by a space, in file/output.')
argparser.add_argument('-s', '--separate', action='store_true', help='If -r is specified, separate prompt and response \
                                                                      by a tab; otherwise ignored.')
argparser.add_argument('-d', '--destination', action='store', nargs='?', help='Destination file to write prompts/responses. \
                                                                               Ignored if files are given.')

args = argparser.parse_args()

chatbot = Chatbot(config={
    'email': args.email,
    'password': args.password,
    'paid': args.plus
})

def prompt_ChatGPT(prompt: str, chatbot: Chatbot):
    response = ''
    for data in chatbot.ask(prompt):
        response = data['message']
    return response

for file in args.files:
    write_file = file.replace('.txt', '_responses.txt')
    if args.retain:
        write_file = file.replace('.txt', '_prompts_and_responses.txt') if not args.separate else file.replace('.txt', '_prompts_and_responses.tsv')
    try: 
        with open(file, 'r') as infile, open(write_file, 'a') as outfile:
            count = 0
            for line in infile.readlines():
                prompt = line.strip()
                response = prompt_ChatGPT(prompt, chatbot)
                line = response if not args.retain else prompt + ' ' + response if not args.separate else prompt + '\t' + response
                outfile.write(line)
                count += 1
            print(f'Wrote {count} lines to {outfile}.\n')
    except:
        print(f'The file {file} could not be opened. Moving onto the next one.\n')

for prompt in args.prompts:
    prompt = prompt.strip()
    response = prompt_ChatGPT(prompt, chatbot)
    line = response if not args.retain else prompt + ' ' + response if not args.separate else prompt + '\t' + response
    print(line, file=args.destination if args.destination else stdout)
