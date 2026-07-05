import { FC } from 'react';

interface LogoProps {
  size?: number;
  className?: string;
}

const Logo: FC<LogoProps> = ({ size = 22, className }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 32 32"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-hidden="true"
  >
    <path
      fill="currentColor"
      d="M16 2 4 6v9c0 7.5 5.2 13.4 12 15 6.8-1.6 12-7.5 12-15V6L16 2Zm0 6 8 2.6V15c0 5.4-3.5 9.8-8 11.2V8Z"
    />
  </svg>
);

export default Logo;
