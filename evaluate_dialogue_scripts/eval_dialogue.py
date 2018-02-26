# Evaluates the dialog output and generates scores based on Will's paper
import string
import numpy as np
import argparse

# file format is 
# dialog X:
# A: blah blah blah
# B: blah blah blah ...
# ...
# dialog X+1:
# A: blah blah blah ...
# B: blah blah blah blah .. 
# and so on...

# run like "python eval_dialogue.py -i sample_output_moose_test.txt"

THRESHOLD_FOR_UNIGRAM_OVERLAP = 0.8

# copied from Brain Huang's code. We can modify this if we want
dull_set = ["I don't know what you're talking about.", "I don't know.", "You don't know.", "You know what I mean.", "I know what you mean.", "You know what I'm saying.", "You don't know anything."]


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", help="Input file to perform analysis on", required=True)

args = parser.parse_args()

input_file = args.input_file


#responses is a 2d array where each inner array contains the entire dialog between two agents
with open(input_file,'r') as f:
	responses = []
	convo = []
	for line in f.readlines():
		if "dialog" in line and len(convo)!=0:
			responses.append(convo)
			convo = []
		elif "dialog" not in line:
			convo.append(line.strip())

def split_dialogue(dialog_seq):
	responses_from_A = []
	responses_from_B = []
	for i in range(0, len(dialog_seq)):
		if i % 2 == 0:
			responses_from_A.append(dialog_seq[i])
		else:
			responses_from_B.append(dialog_seq[i])
	return (responses_from_A, responses_from_B)


# remove "A: " or "B: " from response along with punctuation (using myString.translate(...) trick )
def clean_response_helper(response):
	a_idx = response.find("A: ")
	b_idx = response.find("B: ")
	if a_idx != -1:
		return response[a_idx + 3:].translate(None, string.punctuation)
	if b_idx != -1:
		return response[b_idx + 3:].translate(None, string.punctuation)
	return response.translate(None, string.punctuation)


def unigram_overlap_check(prev_response, next_response):
	words_prev_respose = set(prev_response.split(" "))
	words_next_response = set(next_response.split(" "))
	union = words_next_response.intersection(words_prev_respose)
	# check if more than 80% of the words in the next response overlap with the previous response 
	if len(union)*1.0/len(words_prev_respose) > THRESHOLD_FOR_UNIGRAM_OVERLAP:
		return True
	else:
		return False 

def check_termination_conditions(prev_response, next_response):
	return prev_response in dull_set or unigram_overlap_check(prev_response, next_response)


# unigram diveristy ratio is the total # of words generated in ALL responses by an agent in a given conversation vs.
# the total number of unique words generated
def calculate_unigram_diversity_ratio(all_responses):
	unique_words = set()
	total_num_tokens = 0
	for response in all_responses:
		cleaned_response_split = clean_response_helper(response).split(" ")
		unique_words = unique_words.union(cleaned_response_split)
		total_num_tokens += len(cleaned_response_split)

	return len(unique_words) * 1.0 / total_num_tokens


# given a list of [x1,x2,x3...] generate a new list of all possible bigrams [(x1,x2), (x2,x3)..(x_{n-1}, x_{n})]
def generate_bigrams(some_list):
	bigram_list = []
	for i in range(0, len(some_list) - 1):
		bigram_list.append((some_list[i], some_list[i+1]))
	return bigram_list

def calculate_bigram_diveristy_ratio(all_responses):
	unique_words = set()
	total_num_tokens = 0
	for response in all_responses:
		cleaned_response_split_bigrams = generate_bigrams(clean_response_helper(response).split(" "))
		unique_words = unique_words.union(cleaned_response_split_bigrams)
		total_num_tokens += len(cleaned_response_split_bigrams)

	return len(unique_words) * 1.0 / total_num_tokens


def calculate_dialog_length(all_agent_responses):
	prev_response = all_agent_responses[0]
	turn_counter = 0
	for response in all_agent_responses[1:]:
		next_response = response
		if check_termination_conditions(clean_response_helper(prev_response), clean_response_helper(next_response)):
			break
		turn_counter += 1
	return turn_counter



conversation_lengths = []
unigram_ratios = []
bigram_ratios = []
for conversation in responses:
	responses_A, responses_B = split_dialogue(conversation)
	#print "Length of dialog from agent A is: {}".format(calculate_dialog_length(responses_A))
	#print "Length of dialog from agent B is: {}".format(calculate_dialog_length(responses_B))
	
	# take the minimum of both A and B generated responses
	conversation_lengths.append(min(calculate_dialog_length(responses_A), calculate_dialog_length(responses_B)))

	# get unigram diversity ratios
	unigram_ratios.append(calculate_unigram_diversity_ratio(responses_A))
	unigram_ratios.append(calculate_unigram_diversity_ratio(responses_B))

	# get bigram diversity ratios
	bigram_ratios.append(calculate_bigram_diveristy_ratio(responses_A))
	bigram_ratios.append(calculate_bigram_diveristy_ratio(responses_B))

print "Average # of turns in conversation is: {}".format(np.mean(conversation_lengths))
print "Average unigram diveristy ratio is: {}".format(np.mean(unigram_ratios))
print "Average bigram diversity ratio is:  {}".format(np.mean(bigram_ratios))


