import logging
import torch


def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


def configure_logging(args):
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("i3DMM - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if args.logfile is not None:
        file_logger_handler = logging.FileHandler(args.logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def decode_sdf(decoder_deform, decoder_ref, decoder_col, latent_vec_geom, latent_vec_color, queries, write_debug=False, onlySDFs=False):
    num_samples = queries.shape[0]

    if latent_vec_geom is None:
        inputs = queries
    else:
        latent_repeat_geom = latent_vec_geom.expand(num_samples, -1)
        latent_repeat_geom.requires_grad = False
        latent_repeat_color = latent_vec_color.expand(num_samples, -1)
        latent_repeat_color.requires_grad = False

        inputs_geom = torch.cat([latent_repeat_geom, queries], 1)

    if onlySDFs == True:
        latentxyz = decoder_deform(inputs_geom,write_debug)
        sdf = decoder_ref(queries+latentxyz)
        return sdf, latentxyz

    if write_debug == True:
        sdf = decoder_ref(queries)
        inputs_col = torch.cat([latent_repeat_color, queries], 1)
        color = decoder_col(inputs_col)
        latentxyz = queries
    else:
        latentxyz = decoder_deform(inputs_geom,write_debug)
        sdf = decoder_ref(queries+latentxyz)
        inputs_col = torch.cat([latent_repeat_color, queries+latentxyz], 1)
        color = decoder_col(inputs_col)

    return sdf, color, latentxyz
