from classes.vgg import make_vgg
from classes.extras import make_extras

# vggの動作確認
vgg_test = make_vgg()
print("vgg_test", vgg_test)

# extrasの動作確認
extras_test = make_extras()
print("extras_test", extras_test)
