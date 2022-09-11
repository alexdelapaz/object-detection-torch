import torch
import os
import matplotlib.pyplot as plt
from engine import evaluate
import argparse

# Define analysis function
def run_analysis(artifacts_path):

  # Prior training weights and information
  if os.path.exists(artifacts_path):
      # Stored model information
      model_name = 'FasterRCNN_ResNet50'
      model_path = '{}/{}'.format(artifacts_path,model_name)

      model_info = torch.load(model_path)

      # Prior training run epoch count
      current_epoch = model_info['epochs_trained']
      print('\n-------------------------------------------------------------------------------')
      print('Analysis for |{}: {} epochs of training|'.format(model_name, current_epoch))
      print('-------------------------------------------------------------------------------\n')

      losses_train = model_info['losses_train']
      print('Training loss:\t\t{}'.format(losses_train), end ='\n\n')

      losses_val = model_info['losses_val']
      print('Validation loss:\t{}'.format(losses_val), end ='\n\n')


      # Prior lowest loss for training and validation
      lowest_train_loss = min(losses_train)
      print('Lowest Training loss:\t{}'.format(lowest_train_loss), end ='\n\n')

      lowest_val_loss = min(losses_val)
      print('Lowest Validation loss:\t{}'.format(lowest_val_loss), end ='\n\n')


      evals = model_info['evals']
      if len(evals) > 0:
        print('Final eval data:\n', end ='\n')
        evals[-1].summarize()


      plt.plot(losses_train, color='#00FF00')
      plt.plot(losses_val, color='#4b0082')
      plt.title('FasterRCNN_ResNet50 loss over {} epochs'.format(current_epoch))
      plt.ylabel('Loss')
      plt.xlabel('Epoch')
      plt.xticks(list(range(current_epoch)), list(range(1, current_epoch+1)))
      plt.legend(['Training', 'Validation'], loc='upper left')


      analysis_path = '{}/{}'.format(artifacts_path, 'analysis')
      if not os.path.exists(analysis_path):
        os.mkdir(analysis_path)
      plot_name = 'Loss Graph.png'
      plot_path = '{}/{}'.format(analysis_path, plot_name)
      exp_name = analysis_path.split('/')[-1]

      print('\n-------------------------------------------------------------------------------')
      print('Saving {} artifacts to: {}'.format(exp_name, plot_path), end='\n')
      print('-------------------------------------------------------------------------------\n')
      plt.savefig(plot_path)


if __name__ == '__main__':	
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dir_saved', type = str, required=True, help='Saved model artifacts directory')
	parser.add_argument('-n', '--exp_name', type = str, required=True, help='Experiment name')	

	args_passed = vars(parser.parse_args())

	# Output directory name, default is artifacts
	DIR_OUT = args_passed['dir_saved']
	# Experiment name
	EXP_NAME = args_passed['exp_name']

	PATH = '{}/{}'.format(DIR_OUT,EXP_NAME)
	run_analysis(PATH)