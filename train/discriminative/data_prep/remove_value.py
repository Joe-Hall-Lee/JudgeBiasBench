import argparse
import json
import os

# Training-data root (see README.md, "Training data"). Override with JBB_TRAIN_DATA_DIR.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DISC_DIR = os.path.join(os.environ.get("JBB_TRAIN_DATA_DIR", os.path.join(_REPO_ROOT, "data", "train")), "discriminative")

def filter_bias_data(input_file, output_file):
    """Filters out specified bias types from the input JSONL file."""

    unwanted_bias_types = {'gender', 'race'}
    bias_type_counts = {}

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
                continue

            # Create new lists for filtered data
            new_bias_types = []
            new_rejected = [data['rejected'][0]]  # always keep the original rejected response

            if 'bias_type' in data and 'rejected' in data:
                for i, bias_type in enumerate(data['bias_type']):
                    # Bias types correspond to rejected responses starting from the second one
                    if bias_type not in unwanted_bias_types:
                        new_bias_types.append(bias_type)
                        # i-th bias_type corresponds to (i+1)-th element in 'rejected' list
                        if (i + 1) < len(data['rejected']):
                            new_rejected.append(data['rejected'][i + 1])
                        break

                # Update data with filtered lists
                data['bias_type'] = new_bias_types
                data['rejected'] = new_rejected

            # Count bias types in the modified data
            if 'bias_type' in data:
                bias_type_counts[len(new_bias_types)] = bias_type_counts.get(len(new_bias_types), 0) + 1

            # Write the modified data to the output file
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

    print("\nStatistics of bias types in the cleaned data:")
    for num_bias_types, count in sorted(bias_type_counts.items()):
        print(f"- {num_bias_types} bias types: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove specific bias types from unified feedback data.")
    parser.add_argument("--input_file", type=str, default=os.path.join(DISC_DIR, "unified_feedback_data_4biased_eval.jsonl"),
                        help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, default=os.path.join(DISC_DIR, "unified_feedback_data_4biased_eval_no_value_1biased.jsonl"),
                        help="Path to the output JSONL file.")
    
    args = parser.parse_args()
    
    filter_bias_data(args.input_file, args.output_file)
    print(f"Finished processing. Filtered data saved to {args.output_file}")
