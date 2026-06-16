1. There are two files task_template.py and submission.py onto our cluster inside the tml3 folder.

2. Similarly there are two .sub files on the cluster named rb.sub and submission.sub 

3. First submit the rb.sub job. This will compute the PGD based adversarial training of the model. 

4. This will save the best model according to the highest score calculated. 

5. Then after the model20.pt is created within our cluster in the tml3 folder.

6. We can submit the submission.sub job to upload the model20.pt file from our cluster onto the server where the evaluation takes place to give us the best score on the leaderboard.

