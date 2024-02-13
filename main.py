import argparse

from training import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Experiment setup
    parser.add_argument('--experiment_name', default='exp', help='Name of the experiment')
    parser.add_argument('--dataset_name', default='CUB', help='Name of the dataset')
    parser.add_argument('--dataset_path', default='./data', help='Path to the dataset')
    parser.add_argument('--enable_validation', action='store_true', default=False, help='Enable cross-validation mode')

    # Feature types and preprocessing
    parser.add_argument('--image_embedding_type', default='res101', help='Type of image embedding used (e.g., res101 for features from ResNet-101)')
    parser.add_argument('--attribute_representation_type', default='sent', help='Type of semantic attribute representation (e.g., att for attribute vectors, sent for textual descriptions)')
    parser.add_argument('--enable_preprocessing', type=bool, default=True, help='Enable MinMaxScaler on visual features')
    parser.add_argument('--enable_standardization', action='store_true', default=False, help='Enable feature standardization')

    # Model dimensions
    parser.add_argument('--image_feature_dim', type=int, default=2048, help='Dimension of extracted image features')
    parser.add_argument('--attribute_embedding_size', type=int, default=1024, help='Dimensionality of the semantic attribute embeddings')
    parser.add_argument('--noise_dim', type=int, default=1024, help='Dimension of noise vector for generation')
    parser.add_argument('--embedding_dim', type=int, default=2048, help='Dimension of the embedding layer')
    parser.add_argument('--projected_dim', type=int, default=512, help='Dimension of the projected feature space')
    parser.add_argument('--generator_hidden_dim', type=int, default=4096, help='Dimension of hidden units in generator')
    parser.add_argument('--discriminator_hidden_dim', type=int, default=4096, help='Dimension of hidden units in discriminator')
    parser.add_argument('--comparator_network_dim', type=int, default=2048, help='Dimension of hidden units in comparator network')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2048, help='Input batch size')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--discriminator_train_steps', type=int, default=5, help='Number of discriminator train steps per generator step, following WGAN-GP')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('--classifier_learning_rate', type=float, default=0.001, help='Learning rate for softmax classifier training')
    parser.add_argument('--lr_decay_start_epoch', type=int, default=100, help='Epoch to start applying learning rate decay')
    # parser.add_argument('--lr_decay_rate', type=float, default=0.99, help='Rate of learning rate decay')
    parser.add_argument('--gradient_penalty_weight', type=float, default=10, help='Weight of gradient penalty, following WGAN-GP')
    parser.add_argument('--adam_beta1', type=float, default=0.5, help='Beta1 parameter for Adam optimizer')

    # Loss weights and temperatures
    parser.add_argument('--instance_loss_weight', type=float, default=0.001, help='Weight of instance-level classification loss for generator training')
    parser.add_argument('--class_loss_weight', type=float, default=0.001, help='Weight of class-level score function for generator training')
    parser.add_argument('--instance_temperature', type=float, default=0.1, help='Temperature for instance-level supervision')
    parser.add_argument('--class_temperature', type=float, default=0.1, help='Temperature for class-level supervision')

    # Data and class settings
    parser.add_argument('--num_synthetic_samples', type=int, default=100, help='Number of synthetic features to generate per class')
    parser.add_argument('--total_classes', type=int, default=200, help='Total number of classes')
    parser.add_argument('--seen_classes', type=int, default=150, help='Number of seen classes during training')

    # Miscellaneous
    parser.add_argument('--seed', type=int, default=3483, help='Manual seed for reproducibility')
    parser.add_argument('--resume', help='Path to checkpoint for resuming training')
    parser.add_argument('--turbo', default=False, help='Enable faster training at the cost of more GPU usage')
    parser.add_argument('--early_stopping_patience', default=100, help='Patience for early stopping')

    args = parser.parse_args(args=[])

    if args.dataset_name == 'AWA1': 
        args.attribute_embedding_size = 85
        args.noise_dim = args.attribute_embedding_size
        args.attribute_representation_type = 'att'
        args.seen_classes = 40
        args.total_classes =  50
        args.num_synthetic_samples = 1800
        args.lr_decay_start_epoch = 50

    train(args)