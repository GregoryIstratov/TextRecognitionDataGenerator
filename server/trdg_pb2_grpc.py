# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import trdg_pb2 as trdg__pb2


class TrdgStub(object):
    """Interface exported by the server.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetText = channel.unary_unary(
                '/trdg.Trdg/GetText',
                request_serializer=trdg__pb2.TextRequest.SerializeToString,
                response_deserializer=trdg__pb2.TextData.FromString,
                )


class TrdgServicer(object):
    """Interface exported by the server.
    """

    def GetText(self, request, context):
        """A simple RPC.

        Obtains the feature at a given position.

        A feature with an empty name is returned if there's no feature at the given
        position.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_TrdgServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetText': grpc.unary_unary_rpc_method_handler(
                    servicer.GetText,
                    request_deserializer=trdg__pb2.TextRequest.FromString,
                    response_serializer=trdg__pb2.TextData.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'trdg.Trdg', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Trdg(object):
    """Interface exported by the server.
    """

    @staticmethod
    def GetText(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/trdg.Trdg/GetText',
            trdg__pb2.TextRequest.SerializeToString,
            trdg__pb2.TextData.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
