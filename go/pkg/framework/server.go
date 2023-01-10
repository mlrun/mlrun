// Copyright 2018 Iguazio
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

	"github.com/mlrun/mlrun/proto/build/liveness"

	"github.com/nuclio/errors"
	"github.com/nuclio/loggerus"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/emptypb"
)

type MlrunGRPCServer interface {
	getLogger() *loggerus.Loggerus
	setServer(*grpc.Server)
	getServerOpts() []grpc.ServerOption
	RegisterRoutes(context context.Context)
	OnBeforeStart(context context.Context) error
}

type AbstractMlrunGRPCServer struct {
	Logger         *loggerus.Loggerus
	Server         *grpc.Server
	grpcServerOpts []grpc.ServerOption
}

func NewAbstractMlrunGRPCServer(logger *loggerus.Loggerus, grpcServerOpts []grpc.ServerOption) (*AbstractMlrunGRPCServer, error) {
	return &AbstractMlrunGRPCServer{
		Logger:         logger,
		grpcServerOpts: grpcServerOpts,
	}, nil
}

func (s *AbstractMlrunGRPCServer) getLogger() *loggerus.Loggerus {
	return s.Logger
}

func (s *AbstractMlrunGRPCServer) setServer(server *grpc.Server) {
	s.Server = server
}

func (s *AbstractMlrunGRPCServer) getServerOpts() []grpc.ServerOption {
	return s.grpcServerOpts
}

func (s *AbstractMlrunGRPCServer) RegisterRoutes(ctx context.Context) {
	s.Logger.DebugCtx(ctx, "Registering routes")
	liveness.RegisterReadinessProbeServer(s.Server, s)
	liveness.RegisterLivenessProbeServer(s.Server, s)
}

func (s *AbstractMlrunGRPCServer) OnBeforeStart(ctx context.Context) error {
	s.Logger.DebugCtx(ctx, "Initializing Server")
	return nil
}

func (s *AbstractMlrunGRPCServer) Liveness(context.Context, *emptypb.Empty) (*liveness.LivenessResponse, error) {
	return &liveness.LivenessResponse{Live: true}, nil
}

func (s *AbstractMlrunGRPCServer) Readiness(context.Context, *emptypb.Empty) (*liveness.ReadinessResponse, error) {
	return &liveness.ReadinessResponse{Ready: true}, nil
}

func StartServer(server MlrunGRPCServer, port int) error {
	initContext := context.Background()

	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return errors.Wrap(err, fmt.Sprintf("Failed to listen on port %d", port))
	}

	logger := server.getLogger()
	logger.DebugWithCtx(initContext, "Listening", "port", port)
	grpcServer := grpc.NewServer(server.getServerOpts()...)
	server.setServer(grpcServer)
	server.RegisterRoutes(initContext)

	if err := server.OnBeforeStart(initContext); err != nil {
		return errors.Wrap(err, "Failed running on before start hook")
	}
	logger.DebugWithCtx(initContext, "Starting server")
	if err := grpcServer.Serve(listener); err != nil {
		return errors.Wrap(err, "Failed to start server")
	}

	defer listener.Close()

	return nil
}
