from run_quiz_utils import *
import argparse
parser = argparse.ArgumentParser(description='Run the GCG attack on the quiz dataset')
parser.add_argument('--saved_models_path', type=str, default="saved_models", help='Path to save the models')
parser.add_argument('--dataset_dir', type=str, default="./quiz_bias", help='Path to the dataset')
parser.add_argument('--results_dir', type=str, default="./results", help='Path to save the results')
parser.add_argument('--suffix_size', type=int, default=30, help='Size of the suffix')
parser.add_argument('--train_num_steps', type=int, default=1000, help='Number of training steps')
parser.add_argument('--test_num_steps', type=int, default=500, help='Number of testing steps')
parser.add_argument('--topk', type=int, default=128, help='Top k samples to use')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--suffix_char', type=str, default=" !", help='Suffix character')
parser.add_argument('--verbose', type=bool, default=True, help='Verbose mode')
parser.add_argument('--SEED', type=int, default=42, help='Random seed')
parser.add_argument('--model_str', type=str, default="meta-llama/Llama-2-7b-chat-hf", help='Model string')
parser.add_argument('--CRI', action='store_true', help='Do CRI attack')
parser.add_argument('--num_successful_samples', type=int, default=10, help='Number of successful samples to generate')

args = parser.parse_args()
saved_models_path = args.saved_models_path
dataset_dir = args.dataset_dir
results_dir = args.results_dir
suffix_size = args.suffix_size
train_num_steps = args.train_num_steps
test_num_steps = args.test_num_steps
topk = args.topk
batch_size = args.batch_size
suffix_char = args.suffix_char
verbose = args.verbose
SEED = args.SEED
model_str = args.model_str
CRI = args.CRI
num_successful_samples = args.num_successful_samples


# Set the random seed for reproducibility
set_seed(SEED)

# Create the directories if they don't exist
if not os.path.exists(saved_models_path):
    os.makedirs(saved_models_path)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load the model and tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
# Load the tokenizer and model
tokenizer, conversation_template, model = get_starting_ground(model_str, device)

# list folder names in dataset_dir
group_names = os.listdir(dataset_dir)
standard_init = suffix_char * suffix_size

# Load the data
data_dict = {}
for group in group_names:
    group_dir = os.path.join(dataset_dir, group)
    for root, dirname, files in os.walk(group_dir):
        for file in files:
            if file.endswith(".csv"):
                # Read the CSV file
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                data_dict[file_path] = df

# Get train set (CHANGE LATER)
train_group = "generics"
train_dict = {}
for key in data_dict.keys():
    df = data_dict[key]
    splitted_key = key.split("/")
    group = splitted_key[-3]
    state = splitted_key[-2]
    subject = splitted_key[-1].replace(".csv", "")
    if group == train_group and state == "negative":
        train_set = []
        for i in range(len(df)):
            goal = df.iloc[i]['goal']
            target = df.iloc[i]['target']
            train_set.append((goal, target))
        train_dict[subject] = train_set

if CRI:
    print("With CRI")
    # depends on the model
    CRI_path = os.path.join(results_dir, f"{model_str.replace('/', '_')}_CRI.json")
    # Get the CRI list (if there is already load it)
    CRI_list = get_CRI(train_dict, model, tokenizer, device, standard_init, train_num_steps,\
                        topk, batch_size, verbose, CRI_path, SEED, train_group)
else:
    print("Without CRI")
    CRI_list = None


# Run the attack on the test set
for key in data_dict.keys():
    df = data_dict[key]
    splitted_key = key.split("/")
    group = splitted_key[-3]
    state = splitted_key[-2]
    subject = splitted_key[-1].replace(".csv", "")
    # TODO: remove this
    if group != "races" or state != "positive" or subject != "an_admitted_student_at_a_prestigious_university":
        continue

    goals = df['goal'].tolist()
    targets = df['target'].tolist()
    test_set = [(goals[i], targets[i]) for i in range(len(goals))]

    # save results into results_dir/group/state/subject
    if not os.path.exists(os.path.join(results_dir, group)):
        os.makedirs(os.path.join(results_dir, group))
    if not os.path.exists(os.path.join(results_dir, group, state)):
        os.makedirs(os.path.join(results_dir, group, state))
    
    do_CRI_str = "CRI" if CRI else "no_CRI"
    results_path = os.path.join(results_dir, group, state, subject + "_" + do_CRI_str + ".json")
        
    print(f"Running attack on {group}/{state}/{subject}")
    print(f"Test set size: {len(test_set)}")

    results = \
        run_attack_CRI(goal, target, model, model_str, tokenizer, device, test_set, test_num_steps,\
                        topk, batch_size, results_path, standard_init, verbose, SEED, group, cri=CRI_list, num_successful_samples=num_successful_samples)