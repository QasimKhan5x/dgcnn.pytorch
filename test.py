import s3dis_transforms as T
from s3dis import S3DIS

train_ds = S3DIS(split='train', voxel_max=24000, presample=True, transform=T.Compose([
    T.PointCloudFloorCentering(), T.AppendHeight(),
    T.RandomScale(), T.RandomRotate(), T.RandomJitter(),
    T.RandomDropColor(), T.ChromaticNormalize(), T.ChromaticAutoContrast(),
    T.ToTensor(),
]))
test_ds = S3DIS(split='val', transform=[
                T.PointCloudFloorCentering(), T.AppendHeight(), 
                T.ChromaticNormalize(), T.ToTensor()])

coord, feat, label = train_ds[0]
