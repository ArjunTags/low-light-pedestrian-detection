from datasets import load_dataset

ds = load_dataset("imagefolder", data_dir="/home/arjun/lli-pedestrian/data/llvip_cls")
print(ds)                      # expect train 68256 and test 16604
print(ds["train"].features)    # label names: background, person
ds.push_to_hub("ArjunTags/llvip-person-background-cls")
