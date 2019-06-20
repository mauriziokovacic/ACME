from ACME.utility.FloatTensor import *


def _black():
    return FloatTensor([
            [0,0,0,],
            [1,1,1,],
        ],device='cpu')



def _r():
    return FloatTensor([
            [1,0,0,],
            [1,1,1,],
        ],device='cpu')



def _g():
    return FloatTensor([
            [0,1,0,],
            [1,1,1,],
        ],device='cpu')



def _b():
    return FloatTensor([
            [0,0,1,],
            [1,1,1,],
        ],device='cpu')



def _c():
    return FloatTensor([
            [0,1,1,],
            [1,1,1,],
        ],device='cpu')



def _m():
    return FloatTensor([
            [1,0,1,],
            [1,1,1,],
        ],device='cpu')



def _y():
    return FloatTensor([
            [1,1,0,],
            [1,1,1,],
        ],device='cpu')



def _fire():
    return FloatTensor([
            [0.533333,0,0.082353,],
            [0.929412,0.109804,0.141176,],
            [1,0.498039,0.152941,],
            [1,0.788235,0.054902,],
            [1,0.949020,0,],
            [0.937255,0.894118,0.690196,],
            [1,1,1,],
        ],device='cpu')



def _brown():
    return FloatTensor([
            [0.498039,0.301961,0.243137,],
            [0.721569,0.486275,0.298039,],
            [0.886275,0.713725,0.349020,],
            [0.976471,0.972549,0.443137,],
        ],device='cpu')



def _orange():
    return FloatTensor([
            [0.600000,0.203922,0.015686,],
            [0.850980,0.372549,0.054902,],
            [0.996078,0.600000,0.160784,],
            [0.996078,0.850980,0.556863,],
            [1,1,0.831373,],
        ],device='cpu')



def _blue():
    return FloatTensor([
            [0.015686,0.203922,0.600000,],
            [0.054902,0.372549,0.850980,],
            [0.160784,0.600000,0.996078,],
            [0.556863,0.850980,0.996078,],
            [0.831373,1,1,],
        ],device='cpu')



def _green():
    return FloatTensor([
            [0,0.407843,0.215686,],
            [0.192157,0.639216,0.329412,],
            [0.470588,0.776471,0.474510,],
            [0.760784,0.901961,0.600000,],
            [1,1,0.800000,],
        ],device='cpu')



def _mint():
    return FloatTensor([
            [0.039216,0.219608,0.168627,],
            [0.094118,0.286275,0.196078,],
            [0.192157,0.556863,0.286275,],
            [0.400000,0.768627,0.337255,],
            [0.643137,0.988235,0.807843,],
        ],device='cpu')



def _purple():
    return FloatTensor([
            [0.505882,0.058824,0.486275,],
            [0.533333,0.337255,0.654902,],
            [0.549020,0.588235,0.776471,],
            [0.701961,0.803922,0.890196,],
            [0.929412,0.972549,0.984314,],
        ],device='cpu')



def _sign():
    return FloatTensor([
            [0.019608,0.443137,0.690196,],
            [0.572549,0.772549,0.870588,],
            [0.968627,0.968627,0.968627,],
            [0.956863,0.647059,0.509804,],
            [0.792157,0,0.125490,],
        ],device='cpu')



def _king():
    return FloatTensor([
            [0.101961,0.164706,0.423529,],
            [0.698039,0.121569,0.121569,],
            [0.992157,0.733333,0.176471,],
        ],device='cpu')



def _vision():
    return FloatTensor([
            [0,0,0.274510,],
            [0.109804,0.709804,0.878431,],
        ],device='cpu')



def _turbo():
    return FloatTensor([
            [0.074510,0.313725,0.345098,],
            [0.945098,0.949020,0.709804,],
        ],device='cpu')



def _pinot():
    return FloatTensor([
            [0.094118,0.156863,0.282353,],
            [0.294118,0.423529,0.717647,],
        ],device='cpu')



def _sky():
    return FloatTensor([
            [0.027451,0.396078,0.521569,],
            [1,1,1,],
        ],device='cpu')



def _aqua():
    return FloatTensor([
            [0.074510,0.329412,0.478431,],
            [0.501961,0.815686,0.780392,],
        ],device='cpu')



def _dusk():
    return FloatTensor([
            [0.098039,0.329412,0.482353,],
            [1,0.847059,0.607843,],
        ],device='cpu')



def _relay():
    return FloatTensor([
            [0.227451,0.109804,0.443137,],
            [0.843137,0.427451,0.466667,],
            [1,0.686275,0.482353,],
        ],device='cpu')



def _sweet():
    return FloatTensor([
            [0.247059,0.317647,0.694118,],
            [0.352941,0.333333,0.682353,],
            [0.482353,0.372549,0.674510,],
            [0.560784,0.415686,0.682353,],
            [0.658824,0.415686,0.643137,],
            [0.800000,0.419608,0.556863,],
            [0.945098,0.509804,0.443137,],
            [0.952941,0.643137,0.411765,],
            [0.968627,0.788235,0.470588,],
        ],device='cpu')



def _phoenix():
    return FloatTensor([
            [0.972549,0.211765,0,],
            [0.976471,0.831373,0.137255,],
        ],device='cpu')



def _RYB():
    return FloatTensor([
            [0.996078,0.152941,0.070588,],
            [0.988235,0.376471,0.039216,],
            [0.984314,0.600000,0.007843,],
            [0.988235,0.800000,0.101961,],
            [0.996078,0.996078,0.200000,],
            [0.698039,0.843137,0.196078,],
            [0.400000,0.690196,0.196078,],
            [0.203922,0.486275,0.596078,],
            [0.007843,0.278431,0.996078,],
            [0.266667,0.141176,0.839216,],
            [0.525490,0.003922,0.686275,],
            [0.760784,0.078431,0.376471,],
        ],device='cpu')



def _scale():
    return FloatTensor([
            [0,0,0,],
            [0.498039,0,1,],
            [0,0,1,],
            [0,1,1,],
            [0,1,0,],
            [1,1,0,],
            [1,0.501961,0,],
            [1,0,0,],
        ],device='cpu')



def _paint():
    return FloatTensor([
            [0.929412,0.109804,0.141176,],
            [1,0.949020,0,],
            [0.133333,0.694118,0.298039,],
            [0.070588,0.890196,0.858824,],
            [0,0.635294,0.909804,],
            [0.866667,0.305882,0.701961,],
        ],device='cpu')



def _pastel():
    return FloatTensor([
            [0.729412,0.882353,1,],
            [0.729412,1,0.788235,],
            [1,1,0.729412,],
            [1,0.874510,0.729412,],
            [1,0.701961,0.729412,],
        ],device='cpu')



def _cinolib():
    return FloatTensor([
            [0.992157,0.407843,0.462745,],
            [0.992157,0.529412,0.337255,],
            [0.996078,0.898039,0.615686,],
            [0.776471,0.874510,0.713725,],
            [0.301961,0.756863,0.776471,],
            [0.713725,0.784314,0.901961,],
            [0.486275,0.619608,0.984314,],
            [0.988235,0.349020,0.580392,],
        ],device='cpu')



def _matlab():
    return FloatTensor([
            [0,0.447000,0.741000,],
            [0.850000,0.325000,0.098000,],
            [0.929000,0.694000,0.125000,],
            [0.494000,0.184000,0.556000,],
            [0.466000,0.674000,0.188000,],
            [0.301000,0.745000,0.933000,],
            [0.635000,0.078000,0.184000,],
        ],device='cpu')



def _parula():
    return FloatTensor([
            [0.242200,0.150400,0.660300,],
            [0.250390,0.164995,0.707614,],
            [0.257771,0.181781,0.751138,],
            [0.264729,0.197757,0.795214,],
            [0.270648,0.214676,0.836371,],
            [0.275114,0.234238,0.870986,],
            [0.278300,0.255871,0.899071,],
            [0.280333,0.278233,0.922100,],
            [0.281338,0.300595,0.941376,],
            [0.281014,0.322757,0.957886,],
            [0.279467,0.344671,0.971676,],
            [0.275971,0.366681,0.982905,],
            [0.269914,0.389200,0.990600,],
            [0.260243,0.412329,0.995157,],
            [0.244033,0.435833,0.998833,],
            [0.220643,0.460257,0.997286,],
            [0.196333,0.484719,0.989152,],
            [0.183405,0.507371,0.979795,],
            [0.178643,0.528857,0.968157,],
            [0.176438,0.549905,0.952019,],
            [0.168743,0.570262,0.935871,],
            [0.154000,0.590200,0.921800,],
            [0.146029,0.609119,0.907857,],
            [0.138024,0.627629,0.897290,],
            [0.124814,0.645929,0.888343,],
            [0.111252,0.663500,0.876314,],
            [0.095210,0.679829,0.859781,],
            [0.068871,0.694771,0.839357,],
            [0.029667,0.708167,0.816333,],
            [0.003571,0.720267,0.791700,],
            [0.006657,0.731214,0.766014,],
            [0.043329,0.741095,0.739410,],
            [0.096395,0.750000,0.712038,],
            [0.140771,0.758400,0.684157,],
            [0.171700,0.766962,0.655443,],
            [0.193767,0.775767,0.625100,],
            [0.216086,0.784300,0.592300,],
            [0.246957,0.791795,0.556743,],
            [0.290614,0.797290,0.518829,],
            [0.340643,0.800800,0.478857,],
            [0.390900,0.802871,0.435448,],
            [0.445629,0.802419,0.390919,],
            [0.504400,0.799300,0.348000,],
            [0.561562,0.794233,0.304481,],
            [0.617395,0.787619,0.261238,],
            [0.671986,0.779271,0.222700,],
            [0.724200,0.769843,0.191029,],
            [0.773833,0.759805,0.164610,],
            [0.820314,0.749814,0.153529,],
            [0.863433,0.740600,0.159633,],
            [0.903543,0.733029,0.177414,],
            [0.939257,0.728786,0.209957,],
            [0.972757,0.729771,0.239443,],
            [0.995648,0.743371,0.237148,],
            [0.996986,0.765857,0.219943,],
            [0.995205,0.789252,0.202762,],
            [0.989200,0.813567,0.188533,],
            [0.978629,0.838629,0.176557,],
            [0.967648,0.863900,0.164290,],
            [0.961010,0.889019,0.153676,],
            [0.959671,0.913457,0.142257,],
            [0.962795,0.937338,0.126510,],
            [0.969114,0.960629,0.106362,],
            [0.976900,0.983900,0.080500,],
        ],device='cpu')



def palette(name,device='cuda:0'):
    """
    Returns the color palette with the specified name

    Parameters
    ----------
    name : str
        the name of the color palette
    device : str or torch.device (optional)
        the device the tensor will be stored to (default is 'cuda:0')

    Returns
    -------
    Tensor
        the color tensor

    Raises
    ------
    AssertionError
        if the name is unknown
    """

    color = {
                'black'   : _black,
                'r'       : _r,
                'g'       : _g,
                'b'       : _b,
                'c'       : _c,
                'm'       : _m,
                'y'       : _y,
                'fire'    : _fire,
                'brown'   : _brown,
                'orange'  : _orange,
                'blue'    : _blue,
                'green'   : _green,
                'mint'    : _mint,
                'purple'  : _purple,
                'sign'    : _sign,
                'king'    : _king,
                'vision'  : _vision,
                'turbo'   : _turbo,
                'pinot'   : _pinot,
                'sky'     : _sky,
                'aqua'    : _aqua,
                'dusk'    : _dusk,
                'relay'   : _relay,
                'sweet'   : _sweet,
                'phoenix' : _phoenix,
                'RYB'     : _RYB,
                'scale'   : _scale,
                'paint'   : _paint,
                'pastel'  : _pastel,
                'cinolib' : _cinolib,
                'matlab'  : _matlab,
                'parula'  : _parula,
            }
    if name in color:
        return color[name]().to(device=device)
    assert False, 'Unknown color palette'
