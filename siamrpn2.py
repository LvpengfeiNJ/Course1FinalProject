class SiamRPN2(nn.Module):
  def __init__(self, anchor_num=5):
    super(SiamRPN2, self).__init__()
    self.anchor_num = anchor_num
    self.feature = nn.Sequential(
        # conv1
        nn.Conv2d(3, 192, 11, 2),
        nn.BatchNorm2d(192),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, 2),
        # conv2
        nn.Conv2d(192, 512, 5, 1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, 2),
        # conv3
        nn.Conv2d(512, 768, 3, 1),
        nn.BatchNorm2d(768),
        nn.ReLU(inplace=True),
        # conv4
        nn.Conv2d(768, 768, 3, 1),
        nn.BatchNorm2d(768),
        nn.ReLU(inplace=True),
        # conv5
        nn.Conv2d(768, 512, 3, 1),
        nn.BatchNorm2d(512)
    )

    self.conv_reg_z = nn.Conv2d(512, 512*4*anchor_num, 3, 1)
    self.conv_cls_z = nn.Conv2d(512, 512*2*anchor_num, 3, 1)
    self.conv_reg_x = nn.Conv2d(512, 512, 3)
    self.conv_cls_x = nn.Conv2d(512, 512, 3)
    self.adjust_reg = nn.Conv2d(4*anchor_num, 4*anchor_num, 1)

def forward(self,x):
    return self.feature(x)

    # def forward(self, z, x):
    #   z, x = self.feature(z), self.feature(x)
    #   return z, x
      # return self.inference(x, **self.learn(z))
    
    # def learn(self, z):
    #   z = self.feature(z)
    #   kernel_reg = self.conv_reg_z(z)
    #   kernel_cls = self.conv_cls_z(z)
    #   print("Before view operation, kernel_reg size is:{}, kernel_cls size is:{}".format(kernel_reg, kernel_cls))
    #   k = kernel_reg.size()[-1]
    #   kernel_reg = kernel_reg.view(4*self.anchor_num, 512, k, k)
    #   kernel_cls = kernel_cls.view(2*self.anchor_num, 512, k, k)
    #   print("After view operation,kernel_reg size is:{}, kernel_cls size is:{}".format(kernel_reg, kernel_cls))
    #   print("kernel_reg.size()[-1] is:", k)

    #   return kernel_reg, kernel_cls
    
    # def inference(self, x, kernel_reg, kernel_cls):
    #   x = self.feature(x)
    #   x_reg = self.conv_reg_x(x)
    #   x_cls = self.conv_cls_x(x)
    #   print("Before adjust operation, x_reg size is:{}, x_cls size is:{}".format(x_reg, x_cls))
    #   out_reg = self.adjust_reg(F.conv2d(x_reg, kernel_reg))
    #   out_cls = F.conv2d(x_cls, kernel_cls)
    #   print("After adjust and F.convad() operation, x_reg size is:{}, x_cls size is:{}".format(x_reg, x_cls))

      # return out_reg, out_cls
      
net = SiamRPN2()
# print(net) 
from torchsummary import summary
summary(net, input_size=(3, 127, 127))
# summary(net, input_size=[(3, 127, 127),(3, 256, 256)])