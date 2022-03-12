from ast import arg
import os
import models
import torch
import data_loader
import argparse
import numpy as np
import lib_generation
from PIL import Image
from tqdm import tqdm


from torchvision import transforms
from torch.autograd import Variable
from os import listdir
from os.path import isfile, join
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn import svm
import pickle



def simulation_case1():
  """
  This simulation is for case 1, where cifar10 data is splited into half. The 
  first half is trained as stage 1 and the whole is trained as stage 2, which is
  fully trained.
  """
  torch.manual_seed(0)
  torch.cuda.set_device(0)
  resnet_s0_path = "./pre_trained/resnet_cifar10_s0.pth"
  resnet_s0 = models.ResNet34(num_c=10)
  resnet_s0.load_state_dict(torch.load(resnet_s0_path, map_location="cuda:" + str(0)))
  in_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
  resnet_s0.cuda()
  print("resnet_s0 loaded")

  train_loader, test_loader = data_loader.getTargetDataSet("cifar10", 128, in_transform, "./data")
  trainloader_part1 = []
  trainloader_part2 = []

  for batch_idx, (inputs, targets) in enumerate(train_loader):
      if batch_idx < 150:
          trainloader_part1.append((inputs, targets))
      else:
          trainloader_part2.append((inputs, targets))

  resnet_s0.eval()
  temp_x = torch.rand(2, 3, 32, 32).cuda()
  temp_x = Variable(temp_x)
  temp_list = resnet_s0.feature_list(temp_x)[1]
  num_output = len(temp_list)
  feature_list = np.empty(num_output)
  count = 0
  for out in temp_list:
      feature_list[count] = out.size(1)
      count += 1
  
  lib_generation.compute_performance(
        resnet_s0,
        10,
        feature_list,
        trainloader_part1,
        "resnet_s0_cifar10_train",
    )
  
  lib_generation.compute_performance(
        resnet_s0,
        10,
        feature_list,
        test_loader,
        "resnet_s0_cifar10_test",
    )
  
  sample_mean_s0, precision_s0 = lib_generation.sample_estimator(
        resnet_s0, 10, feature_list, trainloader_part1
    )

  # use ood data to train the ood detector
  out_test_loader = data_loader.getNonTargetDataSet(
                "svhn", 120, in_transform, './data'
            )

  print("ood_loaded")

  for i in tqdm(range(num_output)):
      M_in = lib_generation.get_Mahalanobis_score(
          resnet_s0,
          trainloader_part1,
          10,
          "./output/resnet_cifar10/",
          True,
          "resnet",
          sample_mean_s0,
          precision_s0,
          i,
          0.0,
      )
      M_in = np.asarray(M_in, dtype=np.float32)
      if i == 0:
          Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
      else:
          Mahalanobis_in = np.concatenate(
              (Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1
          )

      M_out = lib_generation.get_Mahalanobis_score(
          resnet_s0,
          out_test_loader,
          10,
          "./output/resnet_cifar10/",
          False,
          "resnet",
          sample_mean_s0,
          precision_s0,
          i,
          0.0,
      )
      M_out = np.asarray(M_out, dtype=np.float32)
      if i == 0:
          Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
      else:
          Mahalanobis_out = np.concatenate(
              (Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1
          )

      M_test = lib_generation.get_Mahalanobis_score(
          resnet_s0,
          test_loader,
          10,
          "./output/resnet_cifar10/",
          False,
          "resnet",
          sample_mean_s0,
          precision_s0,
          i,
          0.0,
      )
      M_test = np.asarray(M_test, dtype=np.float32)
      if i == 0:
          Mahalanobis_test = M_test.reshape((M_test.shape[0], -1))
      else:
          Mahalanobis_test = np.concatenate(
              (Mahalanobis_test, M_test.reshape((M_test.shape[0], -1))), axis=1
          )
      
  Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
  Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
  Mahalanobis_test = np.asarray(Mahalanobis_test, dtype=np.float32)

  print("in and out Mahalanobis computed")

  np.savetxt("./simulation_output/resnet_s0_cifar10_train_m.txt", Mahalanobis_in, delimiter=",")
  print("in Mahalanobis saved")
  np.savetxt("./simulation_output/resnet_s0_cifar10_test_m.txt", Mahalanobis_test, delimiter=",")
  print("test Mahalanobis saved")
  np.savetxt("./simulation_output/resnet_s0_cifar10_out_m.txt", Mahalanobis_out, delimiter=",")
  print("out Mahalanobis saved")


  (
      Mahalanobis_data,
      Mahalanobis_labels,
  ) = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)

  oodd = LogisticRegressionCV()
  oodd.fit(Mahalanobis_data, Mahalanobis_labels)
  print("ood detector s0 trained")
  print(oodd.score(Mahalanobis_data, Mahalanobis_labels))
  pickle.dump(oodd, open("./ood_detector/resnet_s0_cifar10_ood_detector.pkl", "wb"))
  print("ood detector s0 saved")


  """
  Now, load full-trained model
  """

  resnet_s1_path = "./pre_trained/resnet_cifar10_s1.pth"
  resnet_s1 = models.ResNet34(num_c=10)

  resnet_s1.load_state_dict(torch.load(resnet_s1_path, map_location="cuda:" + str(0)))
  in_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
  resnet_s1.cuda()
  print("resnet_s1 loaded")

  train_loader, test_loader = data_loader.getTargetDataSet("cifar10", 128, in_transform, "./data")

  resnet_s1.eval()
  temp_x = torch.rand(2, 3, 32, 32).cuda()
  temp_x = Variable(temp_x)
  temp_list = resnet_s1.feature_list(temp_x)[1]
  num_output = len(temp_list)
  feature_list = np.empty(num_output)
  count = 0
  for out in temp_list:
      feature_list[count] = out.size(1)
      count += 1
  
  lib_generation.compute_performance(
        resnet_s1,
        10,
        feature_list,
        train_loader,
        "resnet_s1_cifar10_train",
    )
  
  lib_generation.compute_performance(
        resnet_s1,
        10,
        feature_list,
        test_loader,
        "resnet_s1_cifar10_test",
    )
  
  sample_mean_s1, precision_s1 = lib_generation.sample_estimator(
        resnet_s1, 10, feature_list, train_loader
    )

  # use ood data to train the ood detector
  out_test_loader = data_loader.getNonTargetDataSet(
                "svhn", 120, in_transform, './data'
            )

  print("ood_loaded")

  for i in tqdm(range(num_output)):
      M_in = lib_generation.get_Mahalanobis_score(
          resnet_s1,
          train_loader,
          10,
          "./output/resnet_cifar10/",
          True,
          "resnet",
          sample_mean_s1,
          precision_s1,
          i,
          0.0,
      )
      M_in = np.asarray(M_in, dtype=np.float32)
      if i == 0:
          Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
      else:
          Mahalanobis_in = np.concatenate(
              (Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1
          )

      M_out = lib_generation.get_Mahalanobis_score(
          resnet_s1,
          out_test_loader,
          10,
          "./output/resnet_cifar10/",
          False,
          "resnet",
          sample_mean_s1,
          precision_s1,
          i,
          0.0,
      )
      M_out = np.asarray(M_out, dtype=np.float32)
      if i == 0:
          Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
      else:
          Mahalanobis_out = np.concatenate(
              (Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1
          )

      M_test = lib_generation.get_Mahalanobis_score(
          resnet_s1,
          test_loader,
          10,
          "./output/resnet_cifar10/",
          False,
          "resnet",
          sample_mean_s1,
          precision_s1,
          i,
          0.0,
      )
      M_test = np.asarray(M_test, dtype=np.float32)
      if i == 0:
          Mahalanobis_test = M_test.reshape((M_test.shape[0], -1))
      else:
          Mahalanobis_test = np.concatenate(
              (Mahalanobis_test, M_test.reshape((M_test.shape[0], -1))), axis=1
          )
      
  Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
  Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
  Mahalanobis_test = np.asarray(Mahalanobis_test, dtype=np.float32)

  print("in and out Mahalanobis computed")

  np.savetxt("./simulation_output/resnet_s1_cifar10_train_m.txt", Mahalanobis_in, delimiter=",")
  print("in Mahalanobis saved")
  np.savetxt("./simulation_output/resnet_s1_cifar10_test_m.txt", Mahalanobis_test, delimiter=",")
  print("test Mahalanobis saved")
  np.savetxt("./simulation_output/resnet_s1_cifar10_out_m.txt", Mahalanobis_out, delimiter=",")
  print("out Mahalanobis saved")


  (
      Mahalanobis_data,
      Mahalanobis_labels,
  ) = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)

  oodd = LogisticRegressionCV()
  oodd.fit(Mahalanobis_data, Mahalanobis_labels)
  print("ood detector s1 trained")
  print(oodd.score(Mahalanobis_data, Mahalanobis_labels))
  pickle.dump(oodd, open("./ood_detector/resnet_s1_cifar10_ood_detector.pkl", "wb"))
  print("ood detector s1 saved")
