# 📁 Dataset Upload Guide for Google Colab

## 🎯 You Only Need ONE File!

For the EMG improvements training, you only need this single file:

```
📄 All_subjects_data.h5 (2.1 GB)
```

**Location on your Mac:**
```
/Users/meghvyas/Desktop/research-paper/Dataset/ULTra-MoCap-processed/All_subjects_data.h5
```

---

## 🚀 Option 1: Upload to Google Drive (Recommended)

This is the BEST option because:
- ✅ Upload once, use many times
- ✅ Automatically accessible in Colab
- ✅ No re-upload needed for each session

### Steps:

1. **Open Google Drive:**
   - Go to https://drive.google.com
   - Sign in with your Google account

2. **Create folder structure:**
   - Click "New" → "Folder"
   - Create: `research-paper`
   - Inside that, create: `Dataset`
   - Inside that, create: `ULTra-MoCap-processed`

   Final path: `/MyDrive/research-paper/Dataset/ULTra-MoCap-processed/`

3. **Upload the dataset:**
   - Navigate to the `ULTra-MoCap-processed` folder in Drive
   - Click "New" → "File upload"
   - Select: `/Users/meghvyas/Desktop/research-paper/Dataset/ULTra-MoCap-processed/All_subjects_data.h5`
   - Wait for upload to complete (~5-15 minutes depending on your internet)

4. **Verify:**
   - Check that the file appears in Drive
   - Size should show as ~2.1 GB

✅ **Done!** The Colab notebook will automatically find it here.

---

## 📤 Option 2: Direct Upload to Colab (Not Recommended)

You CAN upload directly to Colab, but:
- ❌ Need to re-upload for EVERY session
- ❌ Takes 5-15 minutes each time
- ❌ Upload lost if runtime disconnects

Only use this if you can't use Google Drive.

### Steps if you must:

1. In Colab, add a new cell:
```python
from google.colab import files
uploaded = files.upload()
# Then select All_subjects_data.h5 from your computer
```

2. Move file to expected location:
```python
!mkdir -p /content/datasets
!mv All_subjects_data.h5 /content/datasets/
```

3. Update notebook Cell 4 to point to:
```python
H5_PATH = '/content/datasets/All_subjects_data.h5'
```

---

## 🗂️ Other Dataset Files (Optional)

Your dataset folder also contains raw data:

```
Dataset/
├── ULTra-MoCap-processed/
│   └── All_subjects_data.h5        ← ONLY THIS FILE IS NEEDED
│
└── 28751156/                       ← Raw data (not needed for training)
    └── ULTra-MoCap-raw-0/
        ├── P01/ ... P13/           ← Individual CSV files
        └── ...                     (12 GB total)
```

**You DON'T need to upload the `28751156` folder** - it's only raw data that's already processed into the `.h5` file.

---

## ✅ Verification Checklist

Before running Colab notebook:

- [ ] File uploaded to Google Drive
- [ ] Located at: `/MyDrive/research-paper/Dataset/ULTra-MoCap-processed/All_subjects_data.h5`
- [ ] File size is ~2.1 GB
- [ ] Google Drive mounted in Colab (Cell 2)

---

## 🎯 Quick Summary

**What to upload:**
- ✅ `All_subjects_data.h5` (2.1 GB)

**Where to upload:**
- ✅ Google Drive: `/MyDrive/research-paper/Dataset/ULTra-MoCap-processed/`

**What NOT to upload:**
- ❌ `28751156` folder (12 GB - not needed)
- ❌ Individual CSV files (already in the .h5 file)

---

## 💡 Tips

1. **Use Google Drive Upload** (not Colab direct upload)
   - Faster and more reliable
   - Only need to do it once

2. **Check upload completed**
   - Don't start training until upload shows 100%
   - Verify file size in Drive matches local size

3. **Internet speed matters**
   - 2.1 GB file takes 5-15 minutes on typical connection
   - Can take longer on slow internet

4. **Alternative folder structure** (also works):
   ```
   /MyDrive/Dataset/ULTra-MoCap-processed/All_subjects_data.h5
   ```
   The notebook will find it in either location!

---

## 📍 File Locations Reference

**Your Mac:**
```
/Users/meghvyas/Desktop/research-paper/Dataset/ULTra-MoCap-processed/All_subjects_data.h5
```

**Google Drive (after upload):**
```
/MyDrive/research-paper/Dataset/ULTra-MoCap-processed/All_subjects_data.h5
```

**In Colab (auto-detected):**
```
/content/drive/MyDrive/research-paper/Dataset/ULTra-MoCap-processed/All_subjects_data.h5
```

---

## 🆘 Troubleshooting

**"File not found" in Colab:**
- Make sure Drive is mounted (run Cell 2)
- Check file path exactly matches
- Verify upload completed successfully

**Upload stuck/failed:**
- Refresh Google Drive page
- Try uploading again
- Check internet connection
- Try smaller chunks if needed

**Wrong file size:**
- Re-upload the file
- Original should be exactly 2.1 GB (2,251,416,576 bytes)

---

**Ready!** Once uploaded to Drive, you can run the Colab notebook anytime! 🚀
