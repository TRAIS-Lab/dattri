# # train a miniature character-level shakespeare model
# # good for debugging and playing on macbooks and such

# out_dir = 'out-tinystories-train'
# eval_interval = 250 # keep frequent because we'll overfit
# eval_iters = 200
# log_interval = 10 # don't print too too often

# # we expect to overfit on this small dataset, so only save when val improves
# always_save_checkpoint = False

# wandb_log = False # override via command line if you like
# wandb_project = 'tinystories-train'
# wandb_run_name = 'mini-gpt'

# dataset = 'tinystories-train'
# gradient_accumulation_steps = 1
# batch_size = 64
# block_size = 256 # context of up to 256 previous characters

# # baby GPT model :)
# n_layer = 6
# #n_layer = 6
# n_head = 6
# n_embd = 384
# dropout = 0.2

# learning_rate = 1e-3 # with baby networks can afford to go a bit higher
# max_iters = 5000
# lr_decay_iters = 5000 # make equal to max_iters usually
# min_lr = 1e-4 # learning_rate / 10 usually
# beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

# warmup_iters = 100 # not super necessary potentially

# # on macbook also add
# # device = 'cpu'  # run on cpu only
# # compile = False # do not torch compile the model




# ################setting2#################

# # train a miniature character-level shakespeare model
# # good for debugging and playing on macbooks and such

# out_dir = 'out-tinystories-train'
# eval_interval = 250 # keep frequent because we'll overfit
# eval_iters = 200
# log_interval = 10 # don't print too too often

# # we expect to overfit on this small dataset, so only save when val improves
# always_save_checkpoint = False

# wandb_log = False # override via command line if you like
# wandb_project = 'tinystories-train'
# wandb_run_name = 'mini-gpt'

# dataset = 'tinystories-train'
# gradient_accumulation_steps = 1
# batch_size = 64
# block_size = 256 # context of up to 256 previous characters

# # baby GPT model :)
# n_layer = 8
# #n_layer = 6
# n_head = 8
# n_embd = 384
# dropout = 0.2

# learning_rate = 1e-3 # with baby networks can afford to go a bit higher
# max_iters = 35000
# lr_decay_iters = 35000 # make equal to max_iters usually
# min_lr = 1e-4 # learning_rate / 10 usually
# beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

# warmup_iters = 100 # not super necessary potentially

# # on macbook also add
# # device = 'cpu'  # run on cpu only
# # compile = False # do not torch compile the model




################setting3#################

# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-tinystories-train'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'tinystories-train'
wandb_run_name = 'mini-gpt'

dataset = 'tinystories-train'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 8
#n_layer = 6
n_head = 8
n_embd = 384
dropout = 0.2

learning_rate = 5e-4 # with baby networks can afford to go a bit higher
max_iters = 35000
lr_decay_iters = 35000 # make equal to max_iters usually
min_lr = 5e-5 # learning_rate / 10 usually
beta2 = 0.99
# make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model