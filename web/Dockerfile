FROM node:22-alpine AS build
ARG NEXT_PUBLIC_EPOCH_API=""
ENV NEXT_PUBLIC_EPOCH_API=$NEXT_PUBLIC_EPOCH_API
WORKDIR /web
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /web/out /usr/share/nginx/html
