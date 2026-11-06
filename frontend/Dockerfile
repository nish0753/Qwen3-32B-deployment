FROM nginx:alpine
COPY index.html /usr/share/nginx/html/index.html
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY default.conf.template /etc/nginx/templates/default.conf.template
ENV UPSTREAM_HOST=vllm-qwen3-32b:8000
EXPOSE 80
