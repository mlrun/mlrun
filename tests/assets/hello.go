package main

import (
	"github.com/nuclio/nuclio-sdk-go"
)

func Handler(context *nuclio.Context, event nuclio.Event) (interface{}, error) {
	context.Logger.Info("This is an unstrucured %s", "log")

	return nuclio.Response{
		StatusCode:  200,
		ContentType: "application/text",
		Body:        []byte("Hello, from Nuclio :]"),
	}, nil
}
