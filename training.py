import torch
import torch.optim as optim
import itertools
import time
import glob
import os

from losses import SupervisedContrastiveLoss
from datasets import DataLoader
from utils import sample, generate_synthetic_features, calc_gradient_penalty, class_scores_in_matrix, class_scores_for_loop, map_labels
from logger import ExperimentLogger, Result
from models.gan import SyntheticFeatureGenerator, FeatureEmbeddingNet, GANQualityCritic, FeatureAttributeDiscriminator
from models.cls import ZSLClassifier

    
def check_cuda():
    """Checks for CUDA availability and returns the appropriate device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_model(opt, logger, device):
    """Initializes and returns the models based on the experiment opt."""
    models = {
        'generator': SyntheticFeatureGenerator(opt).to(device),
        'embedding_net': FeatureEmbeddingNet(opt).to(device),
        'quality_critic': GANQualityCritic(opt).to(device),
        'discriminator': FeatureAttributeDiscriminator(opt).to(device),
    }
    logger.log_print(f'[INFO]  Models initialized and moved to {device}')
    return models

def initialize_optimizer(models, opt):
    """Initializes and returns optimizers for the models."""
    optimizerD = optim.Adam(
        itertools.chain(
            models['quality_critic'].parameters(), 
            models['embedding_net'].parameters(),
            models['discriminator'].parameters()
        ), 
        lr=opt.learning_rate, betas=(opt.adam_beta1, 0.999)
    )
    optimizerG = optim.Adam(models['generator'].parameters(), lr=opt.learning_rate, betas=(opt.adam_beta1, 0.999))
    return optimizerD, optimizerG

def initialize_generator_values(opt, device):
    """Initializes and return initial values for the generator model"""
    input_feat = torch.FloatTensor(opt.batch_size, opt.image_feature_dim).to(device)
    class_att = torch.FloatTensor(opt.batch_size, opt.attribute_embedding_size).to(device)
    noise_vector = torch.FloatTensor(opt.batch_size, opt.noise_dim).to(device)
    input_label = torch.LongTensor(opt.batch_size).to(device)
    return noise_vector, class_att, input_feat, input_label

def update_discriminator(models, optimizerD, criterion, data, opt, input_features, class_attributes, input_labels, noise):
    """
    Updates the Discriminator network by optimizing the WGAN-GP objective.
    This involves computing the loss for real and fake data, and applying gradient penalty.
    """
    # Enable gradient computation for relevant models
    for model_key in ['quality_critic', 'embedding_net', 'discriminator']:
        for parameter in models[model_key].parameters():
            parameter.requires_grad = True

    discriminator_loss = 0
    for _ in range(opt.discriminator_train_steps):
        sample(data, input_features, class_attributes, input_labels, opt.batch_size)
        models['quality_critic'].zero_grad()
        models['embedding_net'].zero_grad()

        # Process real data
        real_embeddings, real_output = models['embedding_net'](input_features)
        critic_real = models['quality_critic'](input_features, class_attributes).mean()

        # Compute contrastive loss for real instances
        real_contrastive_loss = criterion(real_output, input_labels)

        # Generate fake data
        noise.normal_(0, 1)
        fake_data = models['generator'](noise, class_attributes)
        critic_fake = models['quality_critic'](fake_data.detach(), class_attributes).mean()

        # Calculate gradient penalty and Wasserstein distance
        gradient_penalty = calc_gradient_penalty(models['quality_critic'], input_features, fake_data, class_attributes, opt.batch_size, opt.gradient_penalty_weight)
        wasserstein_distance = critic_real - critic_fake

        # Calculate classification loss
        classification_loss_real = compute_classification_loss(real_embeddings, input_labels, models['discriminator'], data, opt)

        # Total discriminator cost
        discriminator_loss = critic_fake - critic_real + gradient_penalty + real_contrastive_loss + classification_loss_real
        discriminator_loss.backward()
        optimizerD.step()

    return discriminator_loss, wasserstein_distance, real_contrastive_loss, classification_loss_real

def update_generator(models, optimizerG, criterion, data, opt, input_features, class_attributes, input_labels, noise):
    """
    Updates the Generator network by optimizing the WGAN-GP objective.
    This involves generating fake data and computing losses for improving generation quality.
    """
    # Disable gradient computation for networks not being updated in this step
    for model_key in ['quality_critic', 'embedding_net', 'discriminator']:
        for parameter in models[model_key].parameters():
            parameter.requires_grad = False

    # Reset gradients for the generator
    models['generator'].zero_grad()

    # Generate fake data
    noise.normal_(0, 1)  # Populate noise vector with normal distribution
    fake_data = models['generator'](noise, class_attributes)  # Generate fake data based on noise and class attributes

    # Pass fake data through the embedding network to get feature representations
    _, fake_output = models['embedding_net'](fake_data)

    # Compute the loss for the generator based on the quality critic's response
    generator_cost = -models['quality_critic'](fake_data, class_attributes).mean()

    # Contrastive loss to ensure that generated and real data are close in the embedding space
    # First, compute embeddings for real data to use along with fake data embeddings
    real_embeddings, real_output = models['embedding_net'](input_features)

    # Concatenate outputs from real and fake data to compute contrastive loss
    all_outputs = torch.cat((fake_output, real_output.detach()), dim=0)
    fake_contrastive_loss = criterion(all_outputs, torch.cat((input_labels, input_labels), dim=0))

    # Compute classification loss for fake data to ensure it helps in correct classification
    classification_loss_fake = compute_classification_loss(real_embeddings, input_labels, models['discriminator'], data, opt)

    # Calculate total generator error (loss) and backpropagate
    generator_error = generator_cost + opt.instance_loss_weight * fake_contrastive_loss + opt.class_loss_weight * classification_loss_fake
    generator_error.backward()

    # Update generator parameters
    optimizerG.step()

    return generator_error, generator_cost, fake_contrastive_loss, classification_loss_fake

def compute_classification_loss(embeddings, labels, discriminator, data, opt):
    """
    Computes the classification loss either using a loop or a matrix approach based on the 'turbo' option.
    """
    if not opt.turbo:
        return class_scores_for_loop(embeddings, labels, discriminator, data, opt)
    else:
        return class_scores_in_matrix(embeddings, labels, discriminator, data, opt)

def adjust_learning_rate(optimizerD, optimizerG, epoch, opt):
    """
    Adjusts the learning rate for both discriminator and generator optimizers based on the current epoch.
    """
    if (epoch + 1) % opt.lr_decay_start_epoch == 0:
        for param_group in optimizerD.param_groups:
            param_group['lr'] = param_group['lr'] * opt.lr_decay_rate
        for param_group in optimizerG.param_groups:
            param_group['lr'] = param_group['lr'] * opt.lr_decay_rate
            
def perform_zero_shot_learning(models, data, opt, epoch, result_gzsl, result_zsl):
    """
    Performs both generalized and conventional zero-shot learning, updating the results accordingly.
    """
    # Set the generator to evaluation mode for zero-shot learning
    models['generator'].eval()

    # Disable gradient calculations for the embedding network
    for param in models['embedding_net'].parameters():
        param.requires_grad = False

    # Generate synthetic features and labels for zero-shot learning
    synthetic_features, synthetic_labels = generate_synthetic_features(models['generator'], data.unseenclasses, data.attributes, opt.num_synthetic_samples, opt.image_feature_dim, opt.attribute_embedding_size, opt.noise_dim)
    
    # Generalized Zero-Shot Learning (GZSL)
    train_X = torch.cat((data.train_features, synthetic_features), 0)
    train_Y = torch.cat((data.train_labels, synthetic_labels), 0)
    gzsl_classifier = ZSLClassifier(train_X, train_Y, models['embedding_net'], opt.embedding_dim, data, opt.total_classes, opt.classifier_learning_rate, 0.5, 25, opt.num_synthetic_samples, True)
    result_gzsl.update_gzsl(epoch, gzsl_classifier.acc_unseen, gzsl_classifier.acc_seen, gzsl_classifier.h)

    # Conventional Zero-Shot Learning (ZSL)
    mapped_labels = map_labels(synthetic_labels, data.unseenclasses)
    zsl_classifier = ZSLClassifier(synthetic_features, mapped_labels, models['embedding_net'], opt.embedding_dim, data, len(data.unseenclasses), opt.classifier_learning_rate, 0.5, 100, opt.num_synthetic_samples, False)
    result_zsl.update(epoch, zsl_classifier.acc, zsl_classifier.loss)

    return gzsl_classifier, zsl_classifier

def save_best_model(epoch, models, result, model_type, logger, opt):
    """
    Saves the best model for a given zero-shot learning type if the current model is the best.
    """
    if result.save_model:
        pattern = logger.output_directory + '/best_{}_*.tar'.format(model_type)
        for file_path in glob.glob(pattern):
            os.remove(file_path)
        model_path = logger.output_directory + '/best_{}_{}.tar'.format(model_type, epoch)
        logger.save_model(epoch, models, opt.seed, model_path)

def early_stopping_check(epoch, result_gzsl, result_zsl, opt, logger):
    """
    Checks for early stopping conditions based on performance improvements.
    """
    if epoch >= result_gzsl.best_epoch + opt.early_stopping_patience and epoch >= result_zsl.best_epoch + opt.early_stopping_patience:
        logger.log_print('[INFO] Training stopped due to no improvement in accuracy for {} epochs'.format(opt.early_stopping_patience))
        return True
    return False

def reset_embedding_net_gradients(models):
    """
    Re-enables gradient calculations for the embedding network.
    """
    for param in models['embedding_net'].parameters():
        param.requires_grad = True
        
def log_training_progress(logger, epoch, result_zsl, zsl_classifier, result_gzsl, 
                          gzsl_classifier, discriminator_loss, generator_error, 
                          wasserstein_distance, real_contrastive_loss, fake_contrastive_loss, 
                          cls_loss_real, cls_loss_fake, epoch_duration):
    metrics = {
            'epoch': epoch,
            'accuracy': zsl_classifier.acc,
            'loss': zsl_classifier.loss,
            'best_accuracy': result_zsl.best_acc,
            'best_loss': result_zsl.best_loss,
            'best_accuracy_epoch': result_zsl.best_epoch,
            'unseen_accuracy': gzsl_classifier.acc_unseen,
            'seen_accuracy': gzsl_classifier.acc_seen,
            'harmonic_mean': gzsl_classifier.h,
            'best_unseen_accuracy': result_gzsl.best_acc_u,
            'best_seen_accuracy': result_gzsl.best_acc_s,
            'best_harmonic_mean': result_gzsl.best_acc,
            'best_harmonic_mean_epoch': result_gzsl.best_epoch,
            'loss_D': discriminator_loss,
            'loss_G': generator_error,
            'wWasserstein_distance': wasserstein_distance,
            'real_contrastive_loss': real_contrastive_loss,
            'fake_contrastive_loss': fake_contrastive_loss,
            'cls_loss_real': cls_loss_real,
            'cls_loss_fake': cls_loss_fake,
            'time': epoch_duration
        }
    logger.update_history_and_log(metrics)

def train(opt):
    # setup cuda if available
    device = check_cuda()
    
    # logging method
    logger = ExperimentLogger(opt)
    # models initialization
    models = initialize_model(opt, logger, device)
    # resume training
    start_epoch = logger.resume_training(models)
    
    # optimzers
    optimizerD, optimizerG = initialize_optimizer(models, opt)
    # loss function
    criterion = SupervisedContrastiveLoss(opt.instance_temperature).to(device)
    
    # load data and initialize GAN variables
    data = DataLoader(opt)
    noise_vector, class_attributes, input_features, input_labels = initialize_generator_values(opt, device)

    # results for saving the best models
    result_zsl = Result()
    result_gzsl = Result()
    
    # Log experiment start
    logger.start_experiment(data.ntrain, device)
    
    # Main training loop
    start_time = time.time()
    for epoch in range(start_epoch, opt.epochs + 1):
        start_epoch_time = time.time()

        # Training step for each batch
        for i in range(0, data.ntrain, opt.batch_size):
            discriminator_loss, wasserstein_distance, real_contrastive_loss, cls_loss_real = update_discriminator(models, optimizerD, criterion, data, opt, input_features, class_attributes, input_labels, noise_vector)
            generator_error, generator_cost, fake_contrastive_loss, cls_loss_fake = update_generator(models, optimizerG, criterion, data, opt, input_features, class_attributes, input_labels, noise_vector)

        # Adjust learning rate and perform zero-shot learning evaluation
        adjust_learning_rate(optimizerD, optimizerG, epoch, opt)
        gzsl_classifier, zsl_classifier = perform_zero_shot_learning(models, data, opt, epoch, result_gzsl, result_zsl)

        # Calculate and log epoch duration
        epoch_duration = int(time.time() - start_epoch_time)
        log_training_progress(logger, epoch, result_zsl, zsl_classifier, result_gzsl, gzsl_classifier, discriminator_loss, generator_error, wasserstein_distance, real_contrastive_loss, fake_contrastive_loss, cls_loss_real, cls_loss_fake, epoch_duration)

        # Save the best models for GZSL and ZSL
        save_best_model(epoch, models, result_zsl, "zsl", logger, opt)
        save_best_model(epoch, models, result_gzsl, "gzsl", logger, opt)

        # Prepare for the next epoch
        models['generator'].train()  # Reset the generator to training mode
        reset_embedding_net_gradients(models)  # Re-enable gradient calculations for the embedding network

        # Check for early stopping
        if early_stopping_check(epoch, result_gzsl, result_zsl, opt, logger):
            break
        
    # save history
    total_time = int(time.time() - start_time)
    logger.end_experiment(total_time)
    logger.save_csv_history()