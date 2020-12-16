import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import os

from model.config import *
from model.data import *
from model.model import *

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

def train(input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, # input arguments
          target_variable, mask, max_target_len,                                                           # output arguments
          encoder, context_encoder, decoder, embedding,                                                    # network arguments
          encoder_optimizer, context_encoder_optimizer, decoder_optimizer,                                 # optimization arguments
          batch_size, clip, max_length=MAX_LENGTH):                                                        # misc arguments

    # Zero gradients
    encoder_optimizer.zero_grad()
    context_encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    dialog_lengths = dialog_lengths.to(device)
    utt_lengths = utt_lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through utterance encoder
    _, utt_encoder_hidden = encoder(input_variable, utt_lengths)
    
    # Convert utterance encoder final states to batched dialogs for use by context encoder
    context_encoder_input = makeContextEncoderInput(utt_encoder_hidden, dialog_lengths_list, batch_size, batch_indices, dialog_indices)
    
    # Forward pass through context encoder
    context_encoder_outputs, context_encoder_hidden = context_encoder(context_encoder_input, dialog_lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the context encoder's final hidden state
    decoder_hidden = context_encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, context_encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, context_encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    context_encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

def trainIters(voc, pairs, encoder, context_encoder, decoder,
               encoder_optimizer, context_encoder_optimizer, decoder_optimizer, embedding, 
               encoder_n_layers, context_encoder_n_layers, decoder_n_layers, 
               save_dir, n_iteration, batch_size, print_every, clip, corpus_name):
    
    # create a batch iterator for training data
    batch_iterator = batchIterator(voc, pairs, batch_size)
    
    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch, training_dialogs, _, true_batch_size = next(batch_iterator)
        # Extract fields from batch
        input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, _, target_variable, mask, max_target_len = training_batch
        dialog_lengths_list = [len(x) for x in training_dialogs]

        # Run a training iteration with batch
        loss = train(input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, # input arguments
                     target_variable, mask, max_target_len,                                                           # output arguments
                     encoder, context_encoder, decoder, embedding,                                                    # network arguments
                     encoder_optimizer, context_encoder_optimizer, decoder_optimizer,                                 # optimization arguments
                     true_batch_size, clip)                                                                           # misc arguments
        print_loss += loss
        
        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

if __name__ == "__main__":
    # Fix random state
    random.seed(2019)

    # Use the appropriate backend (CUDA if available, CPU otherwise)
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print("Using device", ("cuda" if USE_CUDA else "cpu"))

    # Load vocabulary
    voc = loadPrecomputedVoc(corpus_name, word2index_path, index2word_path)
    # Load unlabeled training data
    train_pairs = loadUnlabeledData(voc, train_path)

    print('Building encoders and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    context_encoder = ContextEncoderRNN(hidden_size, context_encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    # Use appropriate device
    encoder = encoder.to(device)
    context_encoder = context_encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # Compute the number of training iterations we will need in order to achieve the number of epochs specified in the settings at the start of the notebook
    n_iter_per_epoch = len(train_pairs) // batch_size + int(len(train_pairs) % batch_size == 1)
    n_iteration = n_iter_per_epoch * pretrain_epochs

    # Ensure dropout layers are in train mode
    encoder.train()
    context_encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    context_encoder_optimizer = optim.Adam(context_encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    # Run training iterations
    print("Starting Training!")
    print("Will train for {} iterations".format(n_iteration))
    trainIters(voc, train_pairs, encoder, context_encoder, decoder,
               encoder_optimizer, context_encoder_optimizer, decoder_optimizer, embedding, 
               encoder_n_layers, context_encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
               print_every, clip, corpus_name)

    # Save the trained model
    print("Saving!")
    directory = os.path.join(save_dir, corpus_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save({
        'en': encoder.state_dict(),
        'ctx': context_encoder.state_dict(),
        'de': decoder.state_dict(),
        'en_opt': encoder_optimizer.state_dict(),
        'ctx_opt': context_encoder_optimizer.state_dict(),
        'de_opt': decoder_optimizer.state_dict(),
        'voc_dict': voc.__dict__,
        'embedding': embedding.state_dict()
    }, os.path.join(directory, "model.tar"))
