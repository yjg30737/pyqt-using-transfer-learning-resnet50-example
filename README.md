# pyqt-using-transfer-learning-resnet50-example
<div align="center">
  <img src="https://user-images.githubusercontent.com/55078043/229002952-9afe57de-b0b6-400f-9628-b8e0044d3f7b.png" width="150px" height="150px"><br/><br/>
  
  [![](https://dcbadge.vercel.app/api/server/cHekprskVE)](https://discord.gg/cHekprskVE)
</div>

PyQt example of using transfer-learned ResNet50 model to distinguish 7 flowers

## What is the difference between Transfer Learning and Fine Tuning
Note: This is personal opinion by my observation.

If you declare the transform to be applied to a transfer-learned model in a different way such as from

```python
        transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
```

to

```python
        transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.ToTensor(),
        ])
```

the accuracy significantly decreases. For fine-tuned models, as long as the transform is appropriate for the model, the detailed structure does not impact the prediction accuracy.

You don't need code like this that explicitly prevents changes to the weights:

```python
for param in model.parameters():
    param.requires_grad = False
```

Fine-tuning is a more advanced form of transfer learning. However, to understand the mechanism of fine-tuning, consider the following example.

For an example source of transfer learning a ResNet50 model to recognize seven types of flowers specified by the user, please refer to the following <a href="https://www.kaggle.com/code/yoonjunggyu/pytorch-transfer-learning-resnet50/edit">Kaggle notebook</a>.

Also take a look at fine-tuning ResNet50 model example of PyQt: <a href="https://github.com/yjg30737/pyqt-using-finetuned-resnet50-example.git">pyqt-using-finetuned-resnet50-example</a>

## Preview
![image](https://github.com/yjg30737/pyqt-using-transfer-learning-resnet50-example/assets/55078043/b18c087a-a5ee-45b4-b9b5-d85f5f3a2822)
