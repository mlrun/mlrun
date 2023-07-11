// Copyright 2023 Iguazio
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package framework

import (
	"context"
	"fmt"
	"net"
	"runtime/debug"
	"time"

	"github.com/mlrun/mlrun/proto/build/health"

	grpcmiddleware "github.com/grpc-ecosystem/go-grpc-middleware"
	grpcrecovery "github.com/grpc-ecosystem/go-grpc-middleware/recovery"
	"github.com/nuclio/errors"
	"github.com/nuclio/logger"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const HealthWatchInterval = 5 * time.Second

type MlrunGRPCServer interface {

	// RegisterRoutes registers the server's routes
	RegisterRoutes(context context.Context)

	// OnBeforeStart allows to perform operations before the server starts
	OnBeforeStart(context context.Context) error

	// SetServingStatus sets the server's healthcheck serving status
	SetServingStatus(status health.HealthCheckResponse_ServingStatus)

	// setServer sets the server's grpc server
	setServer(*grpc.Server)

	// getServerOpts returns the server's grpc server options
	getServerOpts() []grpc.ServerOption
}

type AbstractMlrunGRPCServer struct {
	Logger         logger.Logger
	Server         *grpc.Server
	servingStatus  health.HealthCheckResponse_ServingStatus
	grpcServerOpts []grpc.ServerOption
}

func NewAbstractMlrunGRPCServer(logger logger.Logger, grpcServerOpts []grpc.ServerOption) (*AbstractMlrunGRPCServer, error) {
	server := &AbstractMlrunGRPCServer{
		Logger: logger.GetChild("grpcserver"),
	}

	// add panic recovery middleware
	grpcServerOpts = append([]grpc.ServerOption{
		grpc.StreamInterceptor(
			grpcmiddleware.ChainStreamServer(
				grpcrecovery.StreamServerInterceptor(
					grpcrecovery.WithRecoveryHandlerContext(server.recoveryHandler),
				),
			),
		),
		grpc.UnaryInterceptor(
			grpcmiddleware.ChainUnaryServer(
				grpcrecovery.UnaryServerInterceptor(
					grpcrecovery.WithRecoveryHandlerContext(server.recoveryHandler),
				),
			),
		),
	}, grpcServerOpts...)

	server.grpcServerOpts = grpcServerOpts
	return server, nil
}

func (s *AbstractMlrunGRPCServer) RegisterRoutes(ctx context.Context) {
	s.Logger.DebugCtx(ctx, "Registering routes")
	health.RegisterHealthServer(s.Server, s)
}

func (s *AbstractMlrunGRPCServer) OnBeforeStart(ctx context.Context) error {
	s.Logger.DebugCtx(ctx, "Initializing Server")
	return nil
}

func (s *AbstractMlrunGRPCServer) Check(context.Context, *health.HealthCheckRequest) (*health.HealthCheckResponse, error) {
	return &health.HealthCheckResponse{
		Status: s.servingStatus,
	}, nil
}

func (s *AbstractMlrunGRPCServer) Watch(request *health.HealthCheckRequest, stream health.Health_WatchServer) error {
	currentStatus := s.servingStatus

	// send once the status
	if err := stream.Send(&health.HealthCheckResponse{
		Status: currentStatus,
	}); err != nil {
		return errors.Wrap(err, "Failed to send initial status")
	}

	// start watching status
	for {
		if s.servingStatus != currentStatus {
			currentStatus = s.servingStatus
			if err := stream.Send(&health.HealthCheckResponse{
				Status: currentStatus,
			}); err != nil {
				return errors.Wrap(err, "Failed to send status")
			}
		}
		time.Sleep(HealthWatchInterval)
	}
}

func StartServer(server MlrunGRPCServer, port int, logger logger.Logger) error {
	initContext := context.Background()

	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return errors.Wrap(err, fmt.Sprintf("Failed to listen on port %d", port))
	}

	logger.DebugWithCtx(initContext, "Listening", "port", port)
	grpcServer := grpc.NewServer(server.getServerOpts()...)
	server.setServer(grpcServer)
	server.RegisterRoutes(initContext)

	if err := server.OnBeforeStart(initContext); err != nil {
		return errors.Wrap(err, "Failed running on before start hook")
	}
	logger.DebugWithCtx(initContext, "Starting server")
	server.SetServingStatus(health.HealthCheckResponse_SERVING)
	defer listener.Close()
	if err := grpcServer.Serve(listener); err != nil {
		return errors.Wrap(err, "Failed to start server")
	}

	return nil
}

func (s *AbstractMlrunGRPCServer) SetServingStatus(status health.HealthCheckResponse_ServingStatus) {
	s.servingStatus = status
}

func (s *AbstractMlrunGRPCServer) getServerOpts() []grpc.ServerOption {
	return s.grpcServerOpts
}

// recoveryHandler is a custom grpc recovery handler that logs the panic and returns an internal error.
// This is used to prevent the server from crashing when a panic occurs. This method is passed to the
// grpcrecovery middleware in the server options.
func (s *AbstractMlrunGRPCServer) recoveryHandler(ctx context.Context, panicInstance interface{}) error {
	s.Logger.ErrorWithCtx(ctx, "Request panicked", "panic", panicInstance, "stack", string(debug.Stack()))
	return status.Errorf(codes.Internal, "%s", panicInstance)
}

func (s *AbstractMlrunGRPCServer) setServer(server *grpc.Server) {
	s.Server = server
}
