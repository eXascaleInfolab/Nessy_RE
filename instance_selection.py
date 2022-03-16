import argparse
import json
import os


def sample_from_data(data, label, head_type, tail_type, path_to_output):
    """Samples instances for binary classification. Selects all positive instances and subsamples
    negative instances that have entities of suitable types."""
    output_data = []
    print(f"Got {len(data)} instances")
    for instance in data:
        if instance["relation"] == label:
            output_data.append(instance)
        else:
            if ((instance["subj_type"] in head_type or head_type == ["ANY"]) and
                    (instance["obj_type"] in tail_type or tail_type == ["ANY"])):
                output_data.append(instance)
    print(f"{len(output_data)} instances were sampled")
    os.makedirs(os.path.dirname(path_to_output), exist_ok=True)
    with open(path_to_output, "w") as f:
        for instance in output_data:
            f.write(json.dumps(instance) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input json")
    parser.add_argument("label", type=str, help="Relation label to sample")
    parser.add_argument("--head_type", nargs='+', help="NER for head entity")
    parser.add_argument("--tail_type", nargs="+", help="NER for tail entity")
    parser.add_argument("output", help="Path to output json")
    args = parser.parse_args()
    with open(args.input) as f:
        data = []
        for line in f:
            instance = json.loads(line)
            if type(instance) == dict:
                data.append(instance)
            elif type(instance) == list:
                data = instance
    sample_from_data(data, args.label, args.head_type, args.tail_type, args.output)


if __name__ == "__main__":
    main()
