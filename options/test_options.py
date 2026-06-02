from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavior during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--skip_save_images', action='store_true', help='skip writing eval images to disk; useful for lightweight tuning')
        parser.add_argument('--eval_metrics', nargs='+', default=['PSNR', 'SSIM'], choices=['PSNR', 'SSIM', 'Foreground-PSNR'], help='image quality metrics to compute during paired evaluation')

        # Compute losses during evaluation
        parser.add_argument("--compute_eval_loss", action="store_true", help="compute and save reconstruction loss during evaluation")

        # rewrite default values
        parser.set_defaults(model='pix2pix')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))

        # evaluation tiling options
        parser.add_argument("--tiled_inference", action="store_true", help="use tiled inference")
        parser.add_argument("--tile_size", type=int, default=256, help="tiled inference size")
        parser.add_argument("--tile_stride", type=int, default=256, help="tiled inference stride")
        self.isTrain = False
        return parser
