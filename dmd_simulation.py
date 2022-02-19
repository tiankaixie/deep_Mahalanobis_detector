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
from sewar.full_ref import uqi

parser = argparse.ArgumentParser(description="PyTorch code: Mahalanobis detector")
parser.add_argument(
    "--batch_size",
    type=int,
    default=200,
    metavar="N",
    help="batch size for data loader",
)
parser.add_argument("--dataset", default="cifar10", help="cifar10 | cifar100 | svhn")
parser.add_argument("--dataroot", default="./data", help="path to dataset")
parser.add_argument("--outf", default="./output/", help="folder to output results")
parser.add_argument("--num_classes", type=int, default=10, help="the # of classes")
parser.add_argument("--net_type", default="resnet", help="resnet | densenet")
parser.add_argument("--gpu", type=int, default=0, help="gpu index")
args = parser.parse_args()


def simulation_cifar10_resnet_imagenet():
    """
    This function simulates trained resnet model on cifar10 dataset, and compute
    1. cifar training data accuracy
    2. cifar testing data accuracy
    3. cifar training data mahalanobis distance
    4. cifar testing data mahalanobis distance
    The results are saved in ./simulation_output/
    _true.txt is for true prediction
    _false.txt is for false prediction
    _m.txt is for mahalanobis distance for each instance
    """
    # set the path to pre-trained model and output
    pre_trained_net = "./pre_trained/" + args.net_type + "_" + args.dataset + ".pth"
    args.outf = args.outf + args.net_type + "_" + args.dataset + "/"
    if os.path.isdir(args.outf) == False:
        os.mkdir(args.outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)
    # check the in-distribution dataset
    if args.dataset == "cifar100":
        args.num_classes = 100
    if args.dataset == "svhn":
        out_dist_list = ["cifar10", "imagenet_resize", "lsun_resize"]
    else:
        out_dist_list = ["imagenet_resize"]

    # load networks
    if args.net_type == "densenet":
        if args.dataset == "svhn":
            model = models.DenseNet3(100, int(args.num_classes))
            model.load_state_dict(
                torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu))
            )
        else:
            model = torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu))
        in_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (125.3 / 255, 123.0 / 255, 113.9 / 255),
                    (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0),
                ),
            ]
        )
    elif args.net_type == "resnet":
        model = models.ResNet34(num_c=args.num_classes)
        model.load_state_dict(
            torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu))
        )
        in_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    model.cuda()
    print("load model: " + args.net_type)

    # load dataset
    print("load target data: ", args.dataset)
    train_loader, test_loader = data_loader.getTargetDataSet(
        args.dataset, args.batch_size, in_transform, args.dataroot
    )

    # set information about feature extaction
    model.eval()
    temp_x = torch.rand(2, 3, 32, 32).cuda()
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print("compute performance training")
    lib_generation.compute_performance(
        model,
        args.num_classes,
        feature_list,
        train_loader,
        args.net_type + "_" + args.dataset + "_train",
    )

    print("compute performance testing")
    lib_generation.compute_performance(
        model,
        args.num_classes,
        feature_list,
        test_loader,
        args.net_type + "_" + args.dataset + "_test",
    )

    print("get sample mean and covariance")
    sample_mean, precision = lib_generation.sample_estimator(
        model, args.num_classes, feature_list, train_loader
    )

    print("get Mahalanobis scores")
    # m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    m_list = [0.0]
    for magnitude in m_list:
        print("Noise: " + str(magnitude))
        for i in range(num_output):
            M_in = lib_generation.get_Mahalanobis_score(
                model,
                test_loader,
                args.num_classes,
                args.outf,
                True,
                args.net_type,
                sample_mean,
                precision,
                i,
                magnitude,
            )
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate(
                    (Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1
                )

            O_in = lib_generation.get_Mahalanobis_score(
                model,
                train_loader,
                args.num_classes,
                args.outf,
                True,
                args.net_type,
                sample_mean,
                precision,
                i,
                magnitude,
            )
            O_in = np.asarray(O_in, dtype=np.float32)
            if i == 0:
                Origin_Mahalanobis_in = O_in.reshape((O_in.shape[0], -1))
            else:
                Origin_Mahalanobis_in = np.concatenate(
                    (Origin_Mahalanobis_in, O_in.reshape((O_in.shape[0], -1))), axis=1
                )
        Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)

        for out_dist in out_dist_list:
            out_test_loader = data_loader.getNonTargetDataSet(
                out_dist, args.batch_size, in_transform, args.dataroot
            )
            print("Out-distribution: " + out_dist)
            for i in range(num_output):
                M_out = lib_generation.get_Mahalanobis_score(
                    model,
                    out_test_loader,
                    args.num_classes,
                    args.outf,
                    False,
                    args.net_type,
                    sample_mean,
                    precision,
                    i,
                    magnitude,
                )
                M_out = np.asarray(M_out, dtype=np.float32)
                if i == 0:
                    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                else:
                    Mahalanobis_out = np.concatenate(
                        (Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1
                    )

        Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)

        test_adv_data = torch.load(
            "./adv_output/resnet_cifar10/"
            + "adv_data_%s_%s_%s.pth" % (args.net_type, args.dataset, "FGSM")
        )

        test_label = torch.load(
            "./adv_output/resnet_cifar10/"
            + "label_%s_%s_%s.pth" % (args.net_type, args.dataset, "FGSM")
        )

        lib_generation.compute_performance_2(
            model,
            args.num_classes,
            feature_list,
            test_adv_data,
            test_label,
            args.net_type + "_" + args.dataset + "_adv",
        )

        for i in range(num_output):
            A_out = lib_generation.get_Mahalanobis_score_adv(
                model,
                test_adv_data,
                test_label,
                args.num_classes,
                "./adv_output/",
                args.net_type,
                sample_mean,
                precision,
                i,
                magnitude,
            )
            A_out = np.asarray(A_out, dtype=np.float32)
            if i == 0:
                Mahalanobis_adv = A_out.reshape((A_out.shape[0], -1))
            else:
                Mahalanobis_adv = np.concatenate(
                    (Mahalanobis_adv, A_out.reshape((A_out.shape[0], -1))), axis=1
                )

    m0 = "./simulation_output/" + args.net_type + "_" + args.dataset + "_train_m.txt"
    np.savetxt(m0, Origin_Mahalanobis_in, delimiter=",")
    m1 = "./simulation_output/" + args.net_type + "_" + args.dataset + "_test_m.txt"
    np.savetxt(m1, Mahalanobis_in, delimiter=",")
    m2 = "./simulation_output/" + args.net_type + "_imagenet_m.txt"
    np.savetxt(m2, Mahalanobis_out, delimiter=",")
    m3 = "./simulation_output/" + args.net_type + "_" + args.dataset + "_adv_m.txt"
    np.savetxt(m3, Mahalanobis_adv, delimiter=",")


def transfer_numpy_to_png():
    print("load target data: cifar10")
    in_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(
            #     (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            # ),
        ]
    )
    train_loader, test_loader = data_loader.getTargetDataSet(
        args.dataset, args.batch_size, in_transform, args.dataroot
    )

    # instance_count = 0
    # for data, _ in tqdm(train_loader, desc="plot training data"):
    #     # print(data.cpu().numpy().shape)
    #     temp = data.cpu().numpy()
    #     for i in tqdm(
    #         range(temp.shape[0]), desc="plot training data part", leave=False
    #     ):
    #         # print(f"instance: {instance_count}")
    #         t = np.array(temp[i])
    #         # print(t.shape)
    #         t = np.transpose(t, (1, 2, 0))
    #         # print(t.shape)
    #         new_im = Image.fromarray((t * 255).astype(np.uint8))
    #         new_im.save("./data_image/cifar10_train_" + str(instance_count) + ".png")
    #         instance_count += 1

    instance_count = 0
    for data, _ in tqdm(test_loader, desc="plot test data"):
        # print(data.cpu().numpy().shape)
        temp = data.cpu().numpy()
        for i in tqdm(range(temp.shape[0]), desc="plot test data part", leave=False):
            # print(f"instance: {instance_count}")
            t = np.array(temp[i])
            # print(t.shape)
            t = np.transpose(t, (1, 2, 0))
            # print(t.shape)
            new_im = Image.fromarray((t * 255).astype(np.uint8))
            new_im.save("./data_image/cifar10_test_" + str(instance_count) + ".png")
            instance_count += 1

    test_adv_data = torch.load(
        "./adv_output/resnet_cifar10/"
        + "adv_data_%s_%s_%s.pth" % (args.net_type, args.dataset, "FGSM")
    )

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    denorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1.0 / s for s in std],
    )
    for i in tqdm(range(test_adv_data.shape[0]), desc="plot adv data"):
        # print(f"instance: {instance_count}")
        t = np.array(test_adv_data[i])
        # print(t.shape)
        # t = np.transpose(t, (1, 2, 0))
        # print(t.shape)

        t = denorm(torch.from_numpy(t))
        new_im = Image.fromarray((np.transpose(t.numpy(), (1, 2, 0)) * 255).astype(np.uint8))
        new_im.save("./data_image/cifar10_test_" + str(instance_count) + ".png")
        instance_count += 1


def train_ood_detector(Mahalanobis_out, Mahalanobis_in):
    magnitude = 0.0
    out_dist = "imagenet"
    (
        Mahalanobis_data,
        Mahalanobis_labels,
    ) = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)
    file_name = os.path.join(
        args.outf,
        "Mahalanobis_%s_%s_%s.npy" % (str(magnitude), args.dataset, out_dist),
    )
    Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
    np.save(file_name, Mahalanobis_data)


def compute_distance_metrix():
    # f1 = open("./distance_matrix/similarity.txt", "w")
    # f2 = open("./distance_matrix/mahalanobis.txt", "w")
    f3 = open("./distance_matrix/image_id_map.txt", "w")
    lookup_path = "./data_image/"
    onlyfiles = [
        f
        for f in listdir(lookup_path)
        if isfile(join(lookup_path, f)) and f.endswith(".png")
    ]
    print(onlyfiles)
    print(f"process similarity of {len(onlyfiles)} images...")
    for i in tqdm(range(len(onlyfiles))):
        # temp_res = []
        # for j in tqdm(range(len(onlyfiles)), leave=False):
        #     if i == j:
        #         temp_res.append(0)
        #     else:
        #         temp_res.append(1 - compute_similarity(lookup_path + onlyfiles[i], lookup_path + onlyfiles[j]))
        f3.write(onlyfiles[i] + "\n")

    # np.save("./distance_matrix/similarity.txt", np.array(res_matrix))
    # print(np.array(res_matrix))

    # f1.close(
    # f2.close()
    f3.close()
    file = open("./distance_matrix/image_id_map.txt", "r")
    content = file.read().split("\n")

    print(content[0])
    file.close()


def quick_select(arr, start, end, k):
    left, right = start, end
    pivot = arr[start + (end - start) // 2]
    while left <= right:
        while left <= right and pivot < arr[left]:
            left += 1
        while left <= right and pivot > arr[right]:
            right -= 1
        if left <= right:
            arr[left], arr[right] = arr[right], arr[left]
            left += 1
            right -= 1
    if start + k - 1 <= right:
        return quick_select(arr, start, right, k)
    if start + k - 1 >= left:
        return quick_select(arr, left, end, k - (left - start))

    return arr[start + k - 1]


if __name__ == "__main__":
    # simulation_cifar10_resnet_imagenet()
    transfer_numpy_to_png()
