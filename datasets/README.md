This folder must contain the datasets needed for NCMF, CMF, HNE baselines, DFMF and DCMF.

#### Instructions
1. Execute the following commands to obtain the datasets for the algorithms in the current directory
```
Download the file from https://drive.google.com/file/d/1AjbgQsBfICeRAEPqW1eXAq7L6jza_tHs/view?usp=sharing and place in the current directory
bzip2 -d datasets_NCMF.tar.bz2
tar -xvf datasets_NCMF.tar
```
2. Ensure the data is available as shown below:
```
(base) user@server:<path to code>/ncmf/datasets$ ll
total 28
drwxrwxr-x 6 user user 4096 Sep  5 23:17 ./
drwxrwxr-x 8 user user 4096 Sep  3 22:11 ../
drwxrwxr-x 5 user user 4096 Sep  1 14:56 CMF/
drwxrwxr-x 5 user user 4096 Sep  1 14:56 DCMF/
drwxrwxr-x 6 user user 4096 Sep  3 09:10 HNE/
drwxrwxr-x 6 user user 4096 Sep  3 23:52 NCMF/
-rw-rw-r-- 1 user user  350 Sep  5 23:17 README.md
```
