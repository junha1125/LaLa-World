Rebuild the OpenVid frame sequences from the original OpenVid-1M release (only needed if you do not use the
pre-extracted frames from `LaLa-World/openvid-frame-sequences-1M`). All outputs go to `<root>/openvid_raw/`.

```bash
R=data/openvid_raw; mkdir -p $R/meta
BASE=https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main
wget -P $R/meta $BASE/data/train/OpenVid-1M.csv $BASE/data/train/OpenVidHD.csv     # 1.2 GB of metadata
python data_preparation/openvid/raw/build_index.py   --root data                    # meta/part_index.jsonl (reads ~141 MB of zip directories)
python data_preparation/openvid/raw/select_clips.py  --root data                    # meta/manifest.csv (filters + download order)
python data_preparation/openvid/raw/fetch_clips.py   --root data --budget-gb 600 --workers 32   # videos/ (resumable; raise the budget to get more)
python data_preparation/openvid/raw/extract_frames.py --root data                   # frames/ + sequences.jsonl (10 frames, 0.5 s apart, 768 px)
python data_preparation/openvid/raw/verify_dataset.py --root data --full
```
