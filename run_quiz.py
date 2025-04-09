from run_quiz_utils import *
import argparse
parser = argparse.ArgumentParser(description='Run the GCG attack on the quiz dataset')
# saved_models_path = "saved_models"
# dataset_dir = "./quiz_bias"
# results_dir = "./results"
# suffix_size = 20
# train_num_steps = 500
# test_num_steps = 500
# early_stop = True
# topk = 16
# batch_size = 16
# suffix_char = " !"
# standard_init = suffix_char * suffix_size
# train_set_size = 25
# verbose = True
# SEED = 42
parser.add_argument('--saved_models_path', type=str, default="saved_models", help='Path to save the models')
parser.add_argument('--dataset_dir', type=str, default="./quiz_bias", help='Path to the dataset')
parser.add_argument('--results_dir', type=str, default="./results", help='Path to save the results')
parser.add_argument('--suffix_size', type=int, default=20, help='Size of the suffix')
parser.add_argument('--train_num_steps', type=int, default=500, help='Number of training steps')
parser.add_argument('--test_num_steps', type=int, default=500, help='Number of testing steps')
parser.add_argument('--early_stop', type=bool, default=True, help='Early stopping')
parser.add_argument('--topk', type=int, default=16, help='Top k samples to use')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--suffix_char', type=str, default=" !", help='Suffix character')
parser.add_argument('--train_set_size', type=int, default=25, help='Size of the training set')
parser.add_argument('--verbose', type=bool, default=True, help='Verbose mode')
parser.add_argument('--SEED', type=int, default=42, help='Random seed')

args = parser.parse_args()
saved_models_path = args.saved_models_path
dataset_dir = args.dataset_dir
results_dir = args.results_dir
suffix_size = args.suffix_size
train_num_steps = args.train_num_steps
test_num_steps = args.test_num_steps
early_stop = args.early_stop
topk = args.topk
batch_size = args.batch_size
suffix_char = args.suffix_char
train_set_size = args.train_set_size
verbose = args.verbose
SEED = args.SEED
# Set the random seed for reproducibility
set_seed(SEED)

# Create the directories if they don't exist
if not os.path.exists(saved_models_path):
    os.makedirs(saved_models_path)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


group_names = ["religions", "ethnicities", "Politicion", "nationalities"]
standard_init = suffix_char * suffix_size

# Load the data
data_dict = {}
for group in group_names:
    group_dir = os.path.join(dataset_dir, group)
    for root, _, files in os.walk(group_dir):
        for file in files:
            if file.endswith(".csv"):
                # Read the CSV file
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                data_dict[file_path] = df
# Get train set (CHANGE LATER)
train_set = []
for key in data_dict.keys():
    df = data_dict[key]
    for i in range(len(df)):
        goal = df.iloc[i]['goal']
        target = df.iloc[i]['target']
        train_set.append((goal, target))
train_set = train_set[:train_set_size]
print(f"Train set size: {len(train_set)}")
# Get test set (CHANGE LATER)
test_set = []
for key in data_dict.keys():
    df = data_dict[key]
    for i in range(len(df)):
        goal = df.iloc[i]['goal']
        target = df.iloc[i]['target']
        test_set.append((goal, target))
test_set = test_set[train_set_size:2*train_set_size]
print(f"Test set size: {len(test_set)}")


# Load the model and tokenizer
model_str = "meta-llama/Llama-2-7b-chat-hf"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
# Load the tokenizer and model
tokenizer, conversation_template, model = get_starting_ground(model_str, device)

# Get the CRI list (if there is already load it)
CRI_list = get_CRI(train_set, model, model_str, tokenizer, device, standard_init, train_num_steps, early_stop, topk, batch_size, verbose)

# Run the attack on the test set
all_test_suffix_list, all_test_loss_list, all_test_success_list, all_test_response_list = \
    run_attack_CRI(goal, target, model, model_str, tokenizer, device, train_set, test_set, num_steps=test_num_steps, early_stop=early_stop, topk=topk, batch_size=batch_size, cri=CRI_list)